/*
 * Copyright (C) 2018 Istituto Italiano di Tecnologia (IIT)
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the
 * GNU Lesser General Public License v2.1 or any later version.
 */

#include "HumanKinematicEstimator.h"
#include "IKWorkerPool.h"
#include "InverseVelocityKinematics/InverseVelocityKinematics.hpp"

#include "IHumanWrench.h"

#include <Wearable/IWear/IWear.h>
#include <iDynTree/InverseKinematics.h>
#include <iDynTree/Model/Model.h>
#include <iDynTree/ModelIO/ModelLoader.h>
#include <yarp/os/LogStream.h>
#include <yarp/os/ResourceFinder.h>
#include <yarp/os/RpcServer.h>
#include <yarp/os/Vocab.h>

#include <iDynTree/Core/EigenHelpers.h>
#include <iDynTree/Model/Traversal.h>

// #include <yarp/sig/Vector.h>
#include <iDynTree/yarp/YARPConversions.h>

#include <BipedalLocomotion/ParametersHandler/YarpImplementation.h>
#include <BipedalLocomotion/FloatingBaseEstimators/LeggedOdometry.h>
#include <BipedalLocomotion/ContactDetectors/SchmittTriggerDetector.h>
#include <BipedalLocomotion/Conversions/matioCppConversions.h>

#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <numeric>

#include "Utils.hpp"

/*!
 * @brief analyze model and list of segments to create all possible segment pairs
 *
 * @param[in] model the full model
 * @param[in] humanSegments list of segments on which look for the possible pairs
 * @param[out] framePairs resulting list of all possible pairs. First element is parent, second is
 * child
 * @param[out] framePairIndeces indeces in the humanSegments list of the pairs in framePairs
 */
static void createEndEffectorsPairs(
    const iDynTree::Model& model,
    std::vector<SegmentInfo>& humanSegments,
    std::vector<std::pair<std::string, std::string>>& framePairs,
    std::vector<std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex>>& framePairIndeces);

static bool getReducedModel(const iDynTree::Model& modelInput,
                            const std::string& parentFrame,
                            const std::string& endEffectorFrame,
                            iDynTree::Model& modelOutput);

const std::string DeviceName = "HumanKinematicEstimator";
const std::string LogPrefix = DeviceName + " :";
constexpr double DefaultPeriod = 0.01;

using namespace hde::devices;
using namespace wearable;

// ==============
// IMPL AND UTILS
// ==============

using ModelJointName = std::string;
using ModelLinkName = std::string;

using WearableLinkName = std::string;
using WearableJointName = std::string;

using InverseVelocityKinematicsSolverName = std::string;

using FloatingBaseName = std::string;

struct WearableJointInfo
{
    WearableJointName name;
    size_t index;
};

// Struct that contains all the data exposed by the HumanState interface
struct SolutionIK
{
    std::vector<double> jointPositions;
    std::vector<double> jointVelocities;

    std::array<double, 3> basePosition;
    std::array<double, 4> baseOrientation;

    std::array<double, 6> baseVelocity;

    std::array<double, 3> CoMPosition;
    std::array<double, 3> CoMVelocity;

    void clear()
    {
        jointPositions.clear();
        jointVelocities.clear();
    }
};

enum SolverIK
{
    global,
    pairwised,
    integrationbased
};

enum rpcCommand
{
    empty,
    calibrate,
    calibrateAll,
    calibrateAllWithWorld,
    calibrateSubTree,
    calibrateRelativeLink,
    setRotationOffset,
    resetCalibration,
};

enum class BaseState
{
    standard, // not direct and not fixed base
    direct,   // use base pose measurements from
    fixed,    // fixed base pose
    external  // base pose estimated by BipedalLocomotionFramework estimator
};

// Container of data coming from the wearable interface
struct WearableStorage
{
    // Maps [model joint / link name] ==> [wearable virtual sensor name]
    //
    // E.g. [Pelvis] ==> [XsensSuit::vLink::Pelvis]. Read from the configuration.
    //
    std::unordered_map<ModelLinkName, WearableLinkName> modelToWearable_LinkName;
    std::unordered_map<ModelJointName, WearableJointInfo> modelToWearable_JointInfo;

    // Maps [wearable virtual sensor name] ==> [virtual sensor]
    std::unordered_map<WearableLinkName, SensorPtr<const sensor::IVirtualLinkKinSensor>>
        linkSensorsMap;
    std::unordered_map<WearableJointName, SensorPtr<const sensor::IVirtualSphericalJointKinSensor>>
        jointSensorsMap;
};

class HumanKinematicEstimator::impl
{
public:
    // Attached interface
    wearable::IWear* iWear = nullptr;
    hde::interfaces::IHumanWrench* iHumanWrench = nullptr;

    bool allowIKFailures;
    bool useXsensJointsAngles;

    float period;
    mutable std::mutex mutex;

    // Rpc
    class CmdParser;
    std::unique_ptr<CmdParser> commandPro;
    yarp::os::RpcServer rpcPort;
    bool applyRpcCommand();

    // Wearable variables
    WearableStorage wearableStorage;

    // Model variables
    iDynTree::Model humanModel;
    FloatingBaseName floatingBaseFrame;

    std::vector<SegmentInfo> segments;
    std::vector<LinkPairInfo> linkPairs;

    // Buffers
    std::unordered_map<std::string, iDynTree::Rotation> linkRotationMatrices;
    std::unordered_map<std::string, iDynTree::Transform> linkTransformMatrices;
    std::unordered_map<std::string, iDynTree::Transform> linkTransformMatricesRaw;
    std::unordered_map<std::string, iDynTree::Rotation> linkOrientationMatrices;
    std::unordered_map<std::string, iDynTree::Twist> linkVelocities;
    iDynTree::VectorDynSize jointConfigurationSolution;
    iDynTree::VectorDynSize jointVelocitiesSolution;
    iDynTree::VectorDynSize jointVelocitiesSolutionFiltered;
    iDynTree::Transform baseTransformSolution;
    iDynTree::Twist baseVelocitySolution;

    iDynTree::Vector3 integralOrientationError;
    iDynTree::Vector3 integralLinearVelocityError;

    std::unordered_map<std::string, iDynTreeHelper::Rotation::rotationDistance>
        linkErrorOrientations;
    std::unordered_map<std::string, iDynTree::Vector3> linkErrorAngularVelocities;

    // IK parameters
    int ikPoolSize{1};
    int maxIterationsIK;
    double costTolerance;
    std::string linearSolverName;
    yarp::os::Value ikPoolOption;
    std::unique_ptr<IKWorkerPool> ikPool;
    SolutionIK solution;
    InverseVelocityKinematicsSolverName inverseVelocityKinematicsSolver;

    double posTargetWeight;
    double rotTargetWeight;
    double linVelTargetWeight;
    double angVelTargetWeight;
    double costRegularization;

    double integrationBasedIKMeasuredLinearVelocityGain;
    double integrationBasedIKMeasuredAngularVelocityGain;
    double integrationBasedIKLinearCorrectionGain;
    double integrationBasedIKAngularCorrectionGain;
    double integrationBasedIKIntegralLinearCorrectionGain;
    double integrationBasedIKIntegralAngularCorrectionGain;
    double integrationBasedJointVelocityLimit;

    std::vector<std::string> custom_jointsVelocityLimitsNames;
    std::vector<iDynTree::JointIndex> custom_jointsVelocityLimitsIndexes;
    iDynTree::VectorDynSize custom_jointsVelocityLimitsValues;
    // Custom Constraint Form: lowerBound<=A*X<=upperBuond
    iDynTree::MatrixDynSize
        customConstraintMatrix; // A, CxN matrix; C: number of Constraints, N: number of
                                // system states: Dofs+6 in floating-based robot
    std::vector<std::string> customConstraintVariables; // X, Nx1  Vector : variables names
    std::vector<iDynTree::JointIndex>
        customConstraintVariablesIndex; // X, Nx1  Vector : variables index
    iDynTree::VectorDynSize customConstraintUpperBound; // upperBuond, Cx1 Vector
    iDynTree::VectorDynSize customConstraintLowerBound; // lowerBound, Cx1 Vector
    iDynTree::VectorDynSize baseVelocityUpperLimit;
    iDynTree::VectorDynSize baseVelocityLowerLimit;
    double k_u, k_l;

    // Secondary calibration
    std::unordered_map<std::string, iDynTree::Rotation> secondaryCalibrationRotations;
    iDynTree::Transform secondaryCalibrationWorld;
    void eraseSecondaryCalibration(const std::string& linkName);
    void selectChainJointsAndLinksForSecondaryCalibration(const std::string& linkName, const std::string& childLinkName,
                                                  std::vector<iDynTree::JointIndex>& jointZeroIndices, std::vector<iDynTree::LinkIndex>& linkToCalibrateIndices);
    void computeSecondaryCalibrationRotationsForChain(const std::vector<iDynTree::JointIndex>& jointZeroIndices, const iDynTree::Transform &refLinkForCalibrationTransform, const std::vector<iDynTree::LinkIndex>& linkToCalibrateIndices, const std::string& refLinkForCalibrationName);

    SolverIK ikSolver;

    // flags
    bool useDirectBaseMeasurement;
    bool useFixedBase;

    iDynTree::InverseKinematics globalIK;
    InverseVelocityKinematics inverseVelocityKinematics;
    iDynTreeHelper::State::integrator stateIntegrator;

    // clock
    double lastTime{-1.0};

    // kinDynComputation
    std::unique_ptr<iDynTree::KinDynComputations> kinDynComputations;
    iDynTree::Vector3 worldGravity;

    // get input data
    bool getJointAnglesFromInputData(iDynTree::VectorDynSize& jointAngles);
    bool getLinkTransformFromInputData(std::unordered_map<std::string, iDynTree::Transform>& t);
    bool getLinkVelocityFromInputData(std::unordered_map<std::string, iDynTree::Twist>& t);

    // calibrate data
    bool applySecondaryCalibration(const std::unordered_map<std::string, iDynTree::Transform>& t_in, std::unordered_map<std::string, iDynTree::Transform>& t_out);

    // solver initialization and update
    bool createLinkPairs();
    bool initializePairwisedInverseKinematicsSolver();
    bool initializeGlobalInverseKinematicsSolver();
    bool initializeIntegrationBasedInverseKinematicsSolver();
    bool solvePairwisedInverseKinematicsSolver();
    bool solveGlobalInverseKinematicsSolver();
    bool solveIntegrationBasedInverseKinematics();

    // optimization targets
    bool updateInverseKinematicTargets();
    bool addInverseKinematicTargets();

    bool updateInverseVelocityKinematicTargets();
    bool addInverseVelocityKinematicsTargets();

    bool computeLinksOrientationErrors(
        std::unordered_map<std::string, iDynTree::Transform> linkDesiredOrientations,
        iDynTree::VectorDynSize jointConfigurations,
        iDynTree::Transform floatingBasePose,
        std::unordered_map<std::string, iDynTreeHelper::Rotation::rotationDistance>&
            linkErrorOrientations);
    bool computeLinksAngularVelocityErrors(
        std::unordered_map<std::string, iDynTree::Twist> linkDesiredVelocities,
        iDynTree::VectorDynSize jointConfigurations,
        iDynTree::Transform floatingBasePose,
        iDynTree::VectorDynSize jointVelocities,
        iDynTree::Twist baseVelocity,
        std::unordered_map<std::string, iDynTree::Vector3>& linkAngularVelocityError);

    // external base estimator and contact detector methods
    bool setupExternalBaseEstimator(yarp::os::Searchable& config);
    bool setupExternalContactDetector(yarp::os::Searchable& config);
    bool initializeEstimatorWorld();
    bool updateExternalEstimatorAndDetector();
    bool logData();

    // external base estimator and contact detector
    std::unique_ptr<BipedalLocomotion::Estimators::LeggedOdometry> extLeggedOdom;
    std::unique_ptr<BipedalLocomotion::Contacts::SchmittTriggerDetector> extSchmitt;
    std::shared_ptr<iDynTree::KinDynComputations> extKinDyn;
    BaseState baseState{BaseState::external};
    bool m_extEstimatorInitialized{false};
    int itrCount{0};
    int initThresh{400};

    double flatContactPlaneInclinationRoll, flatContactPlaneInclinationPitch;
    double flatContactPlaneHeight;
    std::string leftFootString{"LeftFoot"}, rightFootString{"RightFoot"};

    iDynTree::Transform loPose;
    iDynTree::Twist loTwist;

    // log data
    Eigen::VectorXd lfTime, lfContact, rfTime, rfContact, estTime;
    Eigen::VectorXd lfForce, rfForce;
    Eigen::MatrixXd lfWrench, rfWrench;
    Eigen::MatrixXd extPos, extRot, extLinVel, extAngVel;
    Eigen::MatrixXd ikBasePos, ikBaseRot, ikBaseLinVel, ikBaseAngVel;
    Eigen::MatrixXd linkPos, linkRot, linkLinVel, linkAngVel;
    Eigen::MatrixXd outjointPos, outjointVel, outjointVelFilt;
    Eigen::VectorXd fixedFrame;

    // constructor
    impl();
};

// ===============
// RPC PORT PARSER
// ===============

class HumanKinematicEstimator::impl::CmdParser : public yarp::os::PortReader
{

public:
    std::atomic<rpcCommand> cmdStatus{rpcCommand::empty};
    std::string parentLinkName;
    std::string childLinkName;
    std::string refLinkName;
    // variables for manual calibration
    std::atomic<double> roll;  // [deg]
    std::atomic<double> pitch; // [deg]
    std::atomic<double> yaw;   // [deg]

    void resetInternalVariables()
    {
        parentLinkName = "";
        childLinkName = "";
        cmdStatus = rpcCommand::empty;
    }

    bool read(yarp::os::ConnectionReader& connection) override
    {
        yarp::os::Bottle command, response;
        if (command.read(connection)) {
            if (command.get(0).asString() == "help") {
                response.addVocab(yarp::os::Vocab::encode("many"));
                response.addString("The following commands can be used to apply a secondary calibration assuming the subject is in the zero configuration of the model for the calibrated links. \n");
                response.addString("Enter <calibrateAll> to apply a secondary calibration for all the links using the measured base pose \n");
                response.addString("Enter <calibrateAllWithWorld <refLink>> to apply a secondary calibration for all the links assuming the <refLink> to be in the world origin \n");
                response.addString("Enter <calibrate <linkName>> to apply a secondary calibration for the given link \n");
                response.addString("Enter <setRotationOffset <linkName> <r p y [deg]>> to apply a secondary calibration for the given link using the given rotation offset (defined using rpy)\n");
                response.addString("Enter <calibrateSubTree <parentLinkName> <childLinkName>> to apply a secondary calibration for the given chain \n");
                response.addString("Enter <calibrateRelativeLink <parentLinkName> <childLinkName>> to apply a secondary calibration for the child link using the parent link as reference \n");
                response.addString("Enter <reset <linkName>> to remove secondary calibration for the given link \n");
                response.addString("Enter <resetAll> to remove all the secondary calibrations");
            }
            else if (command.get(0).asString() == "calibrateRelativeLink" && !command.get(1).isNull() && !command.get(2).isNull()) {
                this->parentLinkName = command.get(1).asString();
                this->childLinkName = command.get(2).asString();
                response.addString("Entered command <calibrateRelativeLink> is correct, trying to set offset of " + this->childLinkName + " using " + this->parentLinkName + " as reference");
                this->cmdStatus = rpcCommand::calibrateRelativeLink;
            }
            else if (command.get(0).asString() == "calibrateSubTree" && !command.get(1).isNull() && !command.get(2).isNull()) {
                this->parentLinkName = command.get(1).asString();
                this->childLinkName = command.get(2).asString();
                response.addString("Entered command <calibrateSubTree> is correct, trying to set offset for the chain from " + this->parentLinkName + " to " + this->childLinkName);
                this->cmdStatus = rpcCommand::calibrateSubTree;
            }
            else if (command.get(0).asString() == "calibrateAll") {
                this->parentLinkName = "";
                response.addString("Entered command <calibrateAll> is correct, trying to set offset calibration for all the links");
                this->cmdStatus = rpcCommand::calibrateAll;
            }
            else if (command.get(0).asString() == "calibrateAllWithWorld") {
                this->parentLinkName = "";
                this->refLinkName = command.get(1).asString();
                response.addString("Entered command <calibrateAllWithWorld> is correct, trying to set offset calibration for all the links, and setting base link " + this->refLinkName + " to the origin");
                this->cmdStatus = rpcCommand::calibrateAllWithWorld;
            }
            else if (command.get(0).asString() == "calibrate" && !(command.get(1).isNull())) {
                this->parentLinkName = command.get(1).asString();
                response.addString("Entered command <calibrate> is correct, trying to set offset calibration for the link " + this->parentLinkName);
                this->cmdStatus = rpcCommand::calibrate;
            }
            else if (command.get(0).asString() == "setRotationOffset" && !command.get(1).isNull() && command.get(2).isDouble() && command.get(3).isDouble() && command.get(4).isDouble()) {
                this->parentLinkName = command.get(1).asString();
                this->roll = command.get(2).asDouble();
                this->pitch = command.get(3).asDouble();
                this->yaw = command.get(4).asDouble();
                response.addString("Entered command <calibrate> is correct, trying to set rotation offset for the link " + this->parentLinkName);
                this->cmdStatus = rpcCommand::setRotationOffset;
            }
            else if (command.get(0).asString() == "resetAll") {
                response.addString("Entered command <resetAll> is correct, removing all the secondary calibrations ");
                this->cmdStatus = rpcCommand::resetCalibration;
            }
            else if (command.get(0).asString() == "reset" && !command.get(1).isNull()) {
                this->parentLinkName = command.get(1).asString();
                response.addString("Entered command <reset> is correct, trying to remove secondaty calibration for the link " + this->parentLinkName);
                this->cmdStatus = rpcCommand::resetCalibration;
            }
            else {
                response.addString(
                    "Entered command is incorrect. Enter help to know available commands");
            }
        }
        else {
            resetInternalVariables();
            return false;
        }

        yarp::os::ConnectionWriter* reply = connection.getWriter();

        if (reply != NULL) {
            response.write(*reply);
        }
        else
            return false;

        return true;
    }
};

// ===========
// CONSTRUCTOR
// ===========

HumanKinematicEstimator::impl::impl()
    : commandPro(new CmdParser())
{}


// ==============================
// HUMANKINEMATICESTIMATOR DEVICE
// ==============================

HumanKinematicEstimator::HumanKinematicEstimator()
    : PeriodicThread(DefaultPeriod)
    , pImpl{new impl()}
{}

HumanKinematicEstimator::~HumanKinematicEstimator() {}

bool HumanKinematicEstimator::open(yarp::os::Searchable& config)
{
    // ===============================
    // CHECK THE CONFIGURATION OPTIONS
    // ===============================

    if (!(config.check("period") && config.find("period").isFloat64())) {
        yInfo() << LogPrefix << "Using default period:" << DefaultPeriod << "s";
    }

    if (!(config.check("urdf") && config.find("urdf").isString())) {
        yError() << LogPrefix << "urdf option not found or not valid";
        return false;
    }

    if (!(config.check("ikSolver") && config.find("ikSolver").isString())) {
        yError() << LogPrefix << "ikSolver option not found or not valid";
        return false;
    }

    if (!(config.check("useXsensJointsAngles") && config.find("useXsensJointsAngles").isBool())) {
        yError() << LogPrefix << "useXsensJointsAngles option not found or not valid";
        return false;
    }

    std::string baseFrameName;
    if(config.check("floatingBaseFrame") && config.find("floatingBaseFrame").isList() ) {
              baseFrameName = config.find("floatingBaseFrame").asList()->get(0).asString();
              pImpl->useFixedBase = false;
              yWarning() << LogPrefix << "'floatingBaseFrame' configuration option as list is deprecated. Please use a string with the model base name only.";
    }
    else if(config.check("floatingBaseFrame") && config.find("floatingBaseFrame").isString() ) {
              baseFrameName = config.find("floatingBaseFrame").asString();
              pImpl->useFixedBase = false;
    }
    else if(config.check("fixedBaseFrame") && config.find("fixedBaseFrame").isList() ) {
              baseFrameName = config.find("fixedBaseFrame").asList()->get(0).asString();
              pImpl->useFixedBase = true;
              yWarning() << LogPrefix << "'fixedBaseFrame' configuration option as list is deprecated. Please use a string with the model base name only.";
    }
    else if(config.check("fixedBaseFrame") && config.find("fixedBaseFrame").isString() ) {
              baseFrameName = config.find("fixedBaseFrame").asString();
              pImpl->useFixedBase = true;
    }
    else {
        yError() << LogPrefix << "BaseFrame option not found or not valid";
        return false;
    }

    yarp::os::Bottle& linksGroup = config.findGroup("MODEL_TO_DATA_LINK_NAMES");
    if (linksGroup.isNull()) {
        yError() << LogPrefix << "Failed to find group MODEL_TO_DATA_LINK_NAMES";
        return false;
    }
    for (size_t i = 1; i < linksGroup.size(); ++i) {
        if (!(linksGroup.get(i).isList() && linksGroup.get(i).asList()->size() == 2)) {
            yError() << LogPrefix
                     << "Childs of MODEL_TO_DATA_LINK_NAMES must be lists of two elements";
            return false;
        }
        yarp::os::Bottle* list = linksGroup.get(i).asList();
        std::string key = list->get(0).asString();
        yarp::os::Bottle* listContent = list->get(1).asList();

        if (!((listContent->size() == 2) && (listContent->get(0).isString())
              && (listContent->get(1).isString()))) {
            yError() << LogPrefix << "Link list must have two strings";
            return false;
        }
    }

    // =======================================
    // PARSE THE GENERAL CONFIGURATION OPTIONS
    // =======================================

    std::string solverName = config.find("ikSolver").asString();
    if (solverName == "global")
        pImpl->ikSolver = SolverIK::global;
    else if (solverName == "pairwised")
        pImpl->ikSolver = SolverIK::pairwised;
    else if (solverName == "integrationbased")
        pImpl->ikSolver = SolverIK::integrationbased;
    else {
        yError() << LogPrefix << "ikSolver " << solverName << " not found";
        return false;
    }

    pImpl->useXsensJointsAngles = config.find("useXsensJointsAngles").asBool();
    const std::string urdfFileName = config.find("urdf").asString();
    pImpl->floatingBaseFrame = baseFrameName;
    pImpl->period = config.check("period", yarp::os::Value(DefaultPeriod)).asFloat64();

    setPeriod(pImpl->period);

    for (size_t i = 1; i < linksGroup.size(); ++i) {
        yarp::os::Bottle* listContent = linksGroup.get(i).asList()->get(1).asList();

        std::string modelLinkName = listContent->get(0).asString();
        std::string wearableLinkName = listContent->get(1).asString();

        yInfo() << LogPrefix << "Read link map:" << modelLinkName << "==>" << wearableLinkName;
        pImpl->wearableStorage.modelToWearable_LinkName[modelLinkName] = wearableLinkName;
    }

    // ==========================================
    // PARSE THE DEPENDENDT CONFIGURATION OPTIONS
    // ==========================================

    if (pImpl->useXsensJointsAngles) {
        yarp::os::Bottle& jointsGroup = config.findGroup("MODEL_TO_DATA_JOINT_NAMES");
        if (jointsGroup.isNull()) {
            yError() << LogPrefix << "Failed to find group MODEL_TO_DATA_JOINT_NAMES";
            return false;
        }

        for (size_t i = 1; i < jointsGroup.size(); ++i) {
            if (!(jointsGroup.get(i).isList() && jointsGroup.get(i).asList()->size() == 2)) {
                yError() << LogPrefix << "Childs of MODEL_TO_DATA_JOINT_NAMES must be lists";
                return false;
            }
            yarp::os::Bottle* list = jointsGroup.get(i).asList();
            std::string key = list->get(0).asString();
            yarp::os::Bottle* listContent = list->get(1).asList();

            if (!((listContent->size() == 3) && (listContent->get(0).isString())
                  && (listContent->get(1).isString()) && (listContent->get(2).isInt()))) {
                yError() << LogPrefix << "Joint list must have two strings and one integer";
                return false;
            }
        }

        for (size_t i = 1; i < jointsGroup.size(); ++i) {
            yarp::os::Bottle* listContent = jointsGroup.get(i).asList()->get(1).asList();

            std::string modelJointName = listContent->get(0).asString();
            std::string wearableJointName = listContent->get(1).asString();
            size_t wearableJointComponent = listContent->get(2).asInt();

            yInfo() << LogPrefix << "Read joint map:" << modelJointName << "==> ("
                    << wearableJointName << "," << wearableJointComponent << ")";
            pImpl->wearableStorage.modelToWearable_JointInfo[modelJointName] = {
                wearableJointName, wearableJointComponent};
        }
    }

    if (pImpl->ikSolver == SolverIK::pairwised || pImpl->ikSolver == SolverIK::global) {
        if (!(config.check("allowIKFailures") && config.find("allowIKFailures").isBool())) {
            yError() << LogPrefix << "allowFailures option not found or not valid";
            return false;
        }
        if (!(config.check("maxIterationsIK") && config.find("maxIterationsIK").isInt())) {
            yError() << LogPrefix << "maxIterationsIK option not found or not valid";
            return false;
        }

        if (!(config.check("costTolerance") && config.find("costTolerance").isFloat64())) {
            yError() << LogPrefix << "costTolerance option not found or not valid";
            return false;
        }
        if (!(config.check("ikLinearSolver") && config.find("ikLinearSolver").isString())) {
            yError() << LogPrefix << "ikLinearSolver option not found or not valid";
            return false;
        }
        if (!(config.check("posTargetWeight") && config.find("posTargetWeight").isFloat64())) {
            yError() << LogPrefix << "posTargetWeight option not found or not valid";
            return false;
        }

        if (!(config.check("rotTargetWeight") && config.find("rotTargetWeight").isFloat64())) {
            yError() << LogPrefix << "rotTargetWeight option not found or not valid";
            return false;
        }
        if (!(config.check("costRegularization") && config.find("costRegularization").isDouble())) {
            yError() << LogPrefix << "costRegularization option not found or not valid";
            return false;
        }

        pImpl->allowIKFailures = config.find("allowIKFailures").asBool();
        pImpl->maxIterationsIK = config.find("maxIterationsIK").asInt();
        pImpl->costTolerance = config.find("costTolerance").asFloat64();
        pImpl->linearSolverName = config.find("ikLinearSolver").asString();
        pImpl->posTargetWeight = config.find("posTargetWeight").asFloat64();
        pImpl->rotTargetWeight = config.find("rotTargetWeight").asFloat64();
        pImpl->costRegularization = config.find("costRegularization").asDouble();
    }

    if (pImpl->ikSolver == SolverIK::global || pImpl->ikSolver == SolverIK::integrationbased) {
        if (!(config.check("useDirectBaseMeasurement")
              && config.find("useDirectBaseMeasurement").isBool())) {
            yError() << LogPrefix << "useDirectBaseMeasurement option not found or not valid";
            return false;
        }
        if (!(config.check("linVelTargetWeight")
              && config.find("linVelTargetWeight").isFloat64())) {
            yError() << LogPrefix << "linVelTargetWeight option not found or not valid";
            return false;
        }

        if (!(config.check("angVelTargetWeight")
              && config.find("angVelTargetWeight").isFloat64())) {
            yError() << LogPrefix << "angVelTargetWeight option not found or not valid";
            return false;
        }

        if (config.check("inverseVelocityKinematicsSolver")
            && config.find("inverseVelocityKinematicsSolver").isString()) {
            pImpl->inverseVelocityKinematicsSolver =
                config.find("inverseVelocityKinematicsSolver").asString();
        }
        else {
            pImpl->inverseVelocityKinematicsSolver = "moorePenrose";
            yInfo() << LogPrefix << "Using default inverse velocity kinematics solver";
        }

        pImpl->useDirectBaseMeasurement = config.find("useDirectBaseMeasurement").asBool();
        pImpl->linVelTargetWeight = config.find("linVelTargetWeight").asFloat64();
        pImpl->angVelTargetWeight = config.find("angVelTargetWeight").asFloat64();
        pImpl->costRegularization = config.find("costRegularization").asDouble();
    }

    if (pImpl->ikSolver == SolverIK::pairwised) {
        if (!(config.check("ikPoolSizeOption")
              && (config.find("ikPoolSizeOption").isString()
                  || config.find("ikPoolSizeOption").isInt()))) {
            yError() << LogPrefix << "ikPoolOption option not found or not valid";
            return false;
        }

        // Get ikPoolSizeOption
        if (config.find("ikPoolSizeOption").isString()
            && config.find("ikPoolSizeOption").asString() == "auto") {
            yInfo() << LogPrefix << "Using " << std::thread::hardware_concurrency()
                    << " available logical threads for ik pool";
            pImpl->ikPoolSize = static_cast<int>(std::thread::hardware_concurrency());
        }
        else if (config.find("ikPoolSizeOption").isInt()) {
            pImpl->ikPoolSize = config.find("ikPoolSizeOption").asInt();
        }

        // The pairwised IK will always use the measured base pose and velocity for the base link
        if (config.check("useDirectBaseMeasurement")
            && config.find("useDirectBaseMeasurement").isBool()
            && !config.find("useDirectBaseMeasurement").asBool()) {
            yWarning() << LogPrefix
                       << "useDirectBaseMeasurement is required from Pair-Wised IK. Assuming its "
                          "value to be true";
        }
        pImpl->useDirectBaseMeasurement = true;
    }

    if (pImpl->ikSolver == SolverIK::integrationbased) {

        if (!(config.check("integrationBasedIKMeasuredVelocityGainLinRot")
              && config.find("integrationBasedIKMeasuredVelocityGainLinRot").isList()
              && config.find("integrationBasedIKMeasuredVelocityGainLinRot").asList()->size()
                     == 2)) {
            yError()
                << LogPrefix
                << "integrationBasedIKMeasuredVelocityGainLinRot option not found or not valid";
            return false;
        }

        if (!(config.check("integrationBasedIKCorrectionGainsLinRot")
              && config.find("integrationBasedIKCorrectionGainsLinRot").isList()
              && config.find("integrationBasedIKCorrectionGainsLinRot").asList()->size() == 2)) {
            yError() << LogPrefix
                     << "integrationBasedIKCorrectionGainsLinRot option not found or not valid";
            return false;
        }

        if (!(config.check("integrationBasedIKIntegralCorrectionGainsLinRot")
              && config.find("integrationBasedIKIntegralCorrectionGainsLinRot").isList()
              && config.find("integrationBasedIKIntegralCorrectionGainsLinRot").asList()->size()
                     == 2)) {
            yError()
                << LogPrefix
                << "integrationBasedIKIntegralCorrectionGainsLinRot option not found or not valid";
            return false;
        }

        if (config.check("integrationBasedJointVelocityLimit")
            && config.find("integrationBasedJointVelocityLimit").isDouble()) {
            pImpl->integrationBasedJointVelocityLimit =
                config.find("integrationBasedJointVelocityLimit").asDouble();
        }
        else {
            pImpl->integrationBasedJointVelocityLimit =
                1000.0; // if no limits given for a joint we put 1000.0 rad/sec, which is very high
        }

        yarp::os::Bottle* integrationBasedIKMeasuredVelocityGainLinRot =
            config.find("integrationBasedIKMeasuredVelocityGainLinRot").asList();
        yarp::os::Bottle* integrationBasedIKCorrectionGainsLinRot =
            config.find("integrationBasedIKCorrectionGainsLinRot").asList();
        yarp::os::Bottle* integrationBasedIKIntegralCorrectionGainsLinRot =
            config.find("integrationBasedIKIntegralCorrectionGainsLinRot").asList();
        pImpl->integrationBasedIKMeasuredLinearVelocityGain =
            integrationBasedIKMeasuredVelocityGainLinRot->get(0).asFloat64();
        pImpl->integrationBasedIKMeasuredAngularVelocityGain =
            integrationBasedIKMeasuredVelocityGainLinRot->get(0).asFloat64();
        pImpl->integrationBasedIKLinearCorrectionGain =
            integrationBasedIKCorrectionGainsLinRot->get(0).asFloat64();
        pImpl->integrationBasedIKAngularCorrectionGain =
            integrationBasedIKCorrectionGainsLinRot->get(1).asFloat64();
        pImpl->integrationBasedIKIntegralLinearCorrectionGain =
            integrationBasedIKIntegralCorrectionGainsLinRot->get(0).asFloat64();
        pImpl->integrationBasedIKIntegralAngularCorrectionGain =
            integrationBasedIKIntegralCorrectionGainsLinRot->get(1).asFloat64();
    }

    // ===================================
    // PRINT CURRENT CONFIGURATION OPTIONS
    // ===================================

    yInfo() << LogPrefix << "*** ===================================";
    yInfo() << LogPrefix << "*** Period                            :" << pImpl->period;
    yInfo() << LogPrefix << "*** Urdf file name                    :" << urdfFileName;
    yInfo() << LogPrefix << "*** Ik solver                         :" << solverName;
    yInfo() << LogPrefix
            << "*** Use Xsens joint angles            :" << pImpl->useXsensJointsAngles;
    yInfo() << LogPrefix
            << "*** Use Directly base measurement    :" << pImpl->useDirectBaseMeasurement;
    if (pImpl->ikSolver == SolverIK::pairwised || pImpl->ikSolver == SolverIK::global) {
        yInfo() << LogPrefix << "*** Allow IK failures                 :" << pImpl->allowIKFailures;
        yInfo() << LogPrefix << "*** Max IK iterations                 :" << pImpl->maxIterationsIK;
        yInfo() << LogPrefix << "*** Cost Tolerance                    :" << pImpl->costTolerance;
        yInfo() << LogPrefix
                << "*** IK Solver Name                    :" << pImpl->linearSolverName;
        yInfo() << LogPrefix << "*** Position target weight            :" << pImpl->posTargetWeight;
        yInfo() << LogPrefix << "*** Rotation target weight            :" << pImpl->rotTargetWeight;
        yInfo() << LogPrefix
                << "*** Cost regularization              :" << pImpl->costRegularization;
        yInfo() << LogPrefix << "*** Size of thread pool               :" << pImpl->ikPoolSize;
    }
    if (pImpl->ikSolver == SolverIK::integrationbased) {
        yInfo() << LogPrefix << "*** Measured Linear velocity gain     :"
                << pImpl->integrationBasedIKMeasuredLinearVelocityGain;
        yInfo() << LogPrefix << "*** Measured Angular velocity gain    :"
                << pImpl->integrationBasedIKMeasuredAngularVelocityGain;
        yInfo() << LogPrefix << "*** Linear correction gain            :"
                << pImpl->integrationBasedIKLinearCorrectionGain;
        yInfo() << LogPrefix << "*** Angular correction gain           :"
                << pImpl->integrationBasedIKAngularCorrectionGain;
        yInfo() << LogPrefix << "*** Linear integral correction gain   :"
                << pImpl->integrationBasedIKIntegralLinearCorrectionGain;
        yInfo() << LogPrefix << "*** Angular integral correction gain  :"
                << pImpl->integrationBasedIKIntegralAngularCorrectionGain;
        yInfo() << LogPrefix
                << "*** Cost regularization              :" << pImpl->costRegularization;
        yInfo() << LogPrefix << "*** Joint velocity limit             :"
                << pImpl->integrationBasedJointVelocityLimit;
    }
    if (pImpl->ikSolver == SolverIK::integrationbased || pImpl->ikSolver == SolverIK::global) {
        yInfo() << LogPrefix << "*** Inverse Velocity Kinematics solver:"
                << pImpl->inverseVelocityKinematicsSolver;
    }
    yInfo() << LogPrefix << "*** ===================================";

    // ==========================
    // INITIALIZE THE HUMAN MODEL
    // ==========================

    auto& rf = yarp::os::ResourceFinder::getResourceFinderSingleton();
    std::string urdfFilePath = rf.findFile(urdfFileName);
    if (urdfFilePath.empty()) {
        yError() << LogPrefix << "Failed to find file" << config.find("urdf").asString();
        return false;
    }

    std::vector<std::string> jointList{"jL5S1_rotx",
                                       "jRightHip_rotx",
                                       "jLeftHip_rotx" ,"jLeftHip_roty" ,"jLeftHip_rotz",
                                       "jLeftKnee_rotx" ,"jLeftKnee_roty", "jLeftKnee_rotz",
                                       "jLeftAnkle_rotx", "jLeftAnkle_roty", "jLeftAnkle_rotz",
                                       "jLeftBallFoot_rotx", "jLeftBallFoot_roty", "jLeftBallFoot_rotz",
                                       "jRightHip_roty", "jRightHip_rotz",
                                       "jRightKnee_rotx", "jRightKnee_roty", "jRightKnee_rotz",
                                       "jRightAnkle_rotx", "jRightAnkle_roty", "jRightAnkle_rotz",
                                       "jRightBallFoot_rotx", "jRightBallFoot_roty", "jRightBallFoot_rotz",
                                       "jL5S1_roty", "jL5S1_rotz", "jL4L3_rotx", "jL4L3_roty", "jL4L3_rotz",
                                       "jL1T12_rotx", "jL1T12_roty", "jL1T12_rotz",
                                       "jT9T8_rotx", "jT9T8_roty", "jT9T8_rotz",
                                       "jLeftC7Shoulder_rotx", "jT1C7_rotx",
                                       "jRightC7Shoulder_rotx", "jRightC7Shoulder_roty", "jRightC7Shoulder_rotz",
                                       "jRightShoulder_rotx", "jRightShoulder_roty", "jRightShoulder_rotz",
                                       "jRightElbow_rotx", "jRightElbow_roty", "jRightElbow_rotz",
                                       "jRightWrist_rotx", "jRightWrist_roty", "jRightWrist_rotz",
                                       "jT1C7_roty", "jT1C7_rotz",
                                       "jC1Head_rotx", "jC1Head_roty", "jC1Head_rotz",
                                       "jLeftC7Shoulder_roty", "jLeftC7Shoulder_rotz",
                                       "jLeftShoulder_rotx", "jLeftShoulder_roty", "jLeftShoulder_rotz",
                                       "jLeftElbow_rotx", "jLeftElbow_roty", "jLeftElbow_rotz",
                                       "jLeftWrist_rotx", "jLeftWrist_roty", "jLeftWrist_rotz" };

    iDynTree::ModelLoader modelLoader;
    if (!modelLoader.loadReducedModelFromFile(urdfFilePath, jointList) || !modelLoader.isValid()) {
        yError() << LogPrefix << "Failed to load model" << urdfFilePath;
        return false;
    }
    yInfo() << LogPrefix << "----------------------------------------" << modelLoader.isValid();
    yInfo() << LogPrefix << modelLoader.model().toString();
    yInfo() << LogPrefix << modelLoader.model().getNrOfLinks()
            << " , joints: " << modelLoader.model().getNrOfJoints();

    yInfo() << LogPrefix << "base link: "
            << modelLoader.model().getLinkName(modelLoader.model().getDefaultBaseLink());

    // ====================
    // INITIALIZE VARIABLES
    // ====================

    // Get the model from the loader
    pImpl->humanModel = modelLoader.model();

    // Set gravity
    pImpl->worldGravity.zero();
    pImpl->worldGravity(2) = -9.81;

    // Initialize kinDyn computation
    pImpl->kinDynComputations =
        std::unique_ptr<iDynTree::KinDynComputations>(new iDynTree::KinDynComputations());
    pImpl->kinDynComputations->loadRobotModel(modelLoader.model());
    pImpl->kinDynComputations->setFloatingBase(pImpl->floatingBaseFrame);

    pImpl->extKinDyn = std::make_shared<iDynTree::KinDynComputations>();
    pImpl->extKinDyn->loadRobotModel(modelLoader.model());
    pImpl->extKinDyn->setFloatingBase(pImpl->floatingBaseFrame);

    // Initialize World Secondary Calibration
    pImpl->secondaryCalibrationWorld = iDynTree::Transform::Identity();

    // =========================
    // INITIALIZE JOINTS BUFFERS
    // =========================

    // Get the number of joints accordingly to the model
    const size_t nrOfDOFs = pImpl->humanModel.getNrOfDOFs();

    pImpl->solution.jointPositions.resize(nrOfDOFs);
    pImpl->solution.jointVelocities.resize(nrOfDOFs);

    pImpl->jointConfigurationSolution.resize(nrOfDOFs);
    pImpl->jointConfigurationSolution.zero();

    pImpl->jointVelocitiesSolution.resize(nrOfDOFs);
    pImpl->jointVelocitiesSolution.zero();

    // =======================
    // INITIALIZE BASE BUFFERS
    // =======================
    pImpl->baseTransformSolution = iDynTree::Transform::Identity();
    pImpl->baseVelocitySolution.zero();

    // ================================================
    // INITIALIZE CUSTOM CONSTRAINTS FOR INTEGRATION-IK
    // ================================================

    pImpl->customConstraintMatrix.resize(0, 0);
    pImpl->customConstraintVariables.resize(0);
    pImpl->customConstraintLowerBound.resize(0);
    pImpl->customConstraintUpperBound.resize(0);
    pImpl->customConstraintVariablesIndex.resize(0);
    pImpl->custom_jointsVelocityLimitsNames.resize(0);
    pImpl->custom_jointsVelocityLimitsValues.resize(0);
    pImpl->custom_jointsVelocityLimitsIndexes.resize(0);
    pImpl->k_u = 0.5;
    pImpl->k_l = 0.5;

    if (config.check("CUSTOM_CONSTRAINTS")) {

        yarp::os::Bottle& constraintGroup = config.findGroup("CUSTOM_CONSTRAINTS");
        if (constraintGroup.isNull()) {
            yError() << LogPrefix << "Failed to find group CUSTOM_CONSTRAINTS";
            return false;
        }
        if (pImpl->inverseVelocityKinematicsSolver != "QP"
            || pImpl->ikSolver != SolverIK::integrationbased) {
            yWarning()
                << LogPrefix
                << "'CUSTOM_CONSTRAINTS' group option is available only if "
                   "'ikSolver==integrationbased' & 'inverseVelocityKinematicsSolver==QP'. \n "
                   "Currently, you are NOT using the customized constraint group.";
        }

        yInfo() << "==================>>>>>>> constraint group: " << constraintGroup.size();

        for (size_t i = 1; i < constraintGroup.size(); i++) {
            yInfo() << "group " << i;
            if (!(constraintGroup.get(i).isList()
                  && constraintGroup.get(i).asList()->size() == 2)) {
                yError() << LogPrefix
                         << "Childs of CUSTOM_CONSTRAINTS must be lists of two elements";
                return false;
            }
            else {
                yInfo() << "Everything is fine...";
            }
            yarp::os::Bottle* constraintList = constraintGroup.get(i).asList();
            std::string constraintKey = constraintList->get(0).asString();
            yarp::os::Bottle* constraintListContent = constraintList->get(1).asList();
            yInfo() << constraintKey;
            if (constraintKey == "custom_joints_velocity_limits_names") {

                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    pImpl->custom_jointsVelocityLimitsNames.push_back(
                        constraintListContent->get(i).asString());
                }
                yInfo() << "custom_joints_velocity_limits_names: ";
                for (size_t i = 0; i < pImpl->custom_jointsVelocityLimitsNames.size(); i++) {
                    yInfo() << pImpl->custom_jointsVelocityLimitsNames[i];
                }
            } // another option
            else if (constraintKey == "custom_joints_velocity_limits_values") {
                pImpl->custom_jointsVelocityLimitsValues.resize(constraintListContent->size());
                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    pImpl->custom_jointsVelocityLimitsValues.setVal(
                        i, constraintListContent->get(i).asDouble());
                }
                yInfo() << "custom_joints_velocity_limits_values: ";
                for (size_t i = 0; i < pImpl->custom_jointsVelocityLimitsValues.size(); i++) {
                    yInfo() << pImpl->custom_jointsVelocityLimitsValues.getVal(i);
                }
            } // another option
            else if (constraintKey == "custom_constraint_variables") {

                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    pImpl->customConstraintVariables.push_back(
                        constraintListContent->get(i).asString());
                }
                yInfo() << "custom_constraint_variables: ";
                for (size_t i = 0; i < pImpl->customConstraintVariables.size(); i++) {
                    yInfo() << pImpl->customConstraintVariables[i];
                }
            } // another option
            else if (constraintKey == "custom_constraint_matrix") {
                // pImpl->customConstraintMatrix.resize(constraintListContent->size());
                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    yarp::os::Bottle* innerLoop = constraintListContent->get(i).asList();
                    if (i == 0) {
                        pImpl->customConstraintMatrix.resize(constraintListContent->size(),
                                                             innerLoop->size());
                    }
                    for (size_t j = 0; j < innerLoop->size(); j++) {
                        pImpl->customConstraintMatrix.setVal(i, j, innerLoop->get(j).asDouble());
                    }
                }
                yInfo() << "Constraint matrix: ";
                for (size_t i = 0; i < pImpl->customConstraintMatrix.rows(); i++) {
                    for (size_t j = 0; j < pImpl->customConstraintMatrix.cols(); j++) {
                        std::cout << pImpl->customConstraintMatrix.getVal(i, j) << " ";
                    }
                    std::cout << std::endl;
                }
            } // another option
            else if (constraintKey == "custom_constraint_upper_bound") {

                pImpl->customConstraintUpperBound.resize(constraintListContent->size());
                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    pImpl->customConstraintUpperBound.setVal(
                        i, constraintListContent->get(i).asDouble());
                }
                yInfo() << "custom_constraint_upper_bound: ";
                for (size_t i = 0; i < pImpl->customConstraintUpperBound.size(); i++) {
                    yInfo() << pImpl->customConstraintUpperBound.getVal(i);
                }
            } // another option
            else if (constraintKey == "custom_constraint_lower_bound") {
                pImpl->customConstraintLowerBound.resize(constraintListContent->size());
                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    pImpl->customConstraintLowerBound.setVal(
                        i, constraintListContent->get(i).asDouble());
                }
                yInfo() << "custom_constraint_lower_bound: ";
                for (size_t i = 0; i < pImpl->customConstraintLowerBound.size(); i++) {
                    yInfo() << pImpl->customConstraintLowerBound.getVal(i);
                }
            } // another option
            else if (constraintKey == "base_velocity_limit_upper_buond") {
                if (constraintListContent->size() != 6) {
                    yError() << "the base velocity limit should have size of 6.";
                    return false;
                }
                pImpl->baseVelocityUpperLimit.resize(6);
                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    pImpl->baseVelocityUpperLimit.setVal(i,
                                                         constraintListContent->get(i).asDouble());
                }
                yInfo() << "base_velocity_limit_upper_buond: ";
                for (size_t i = 0; i < pImpl->baseVelocityUpperLimit.size(); i++) {
                    yInfo() << pImpl->baseVelocityUpperLimit.getVal(i);
                }
            } // another option
            else if (constraintKey == "base_velocity_limit_lower_buond") {
                if (constraintListContent->size() != 6) {
                    yError() << "the base velocity limit should have size of 6.";
                    return false;
                }
                pImpl->baseVelocityLowerLimit.resize(6);
                for (size_t i = 0; i < constraintListContent->size(); i++) {
                    pImpl->baseVelocityLowerLimit.setVal(i,
                                                         constraintListContent->get(i).asDouble());
                }
                yInfo() << "base_velocity_limit_lower_buond: ";
                for (size_t i = 0; i < pImpl->baseVelocityLowerLimit.size(); i++) {
                    yInfo() << pImpl->baseVelocityLowerLimit.getVal(i);
                }
            } // another option
            else if (constraintKey == "k_u") {
                if (constraintGroup.check("k_u") && constraintGroup.find("k_u").isDouble()) {
                    pImpl->k_u = constraintGroup.find("k_u").asDouble();
                    yInfo() << "k_u: " << pImpl->k_u;
                }
            } // another option
            else if (constraintKey == "k_l") {
                if (constraintGroup.check("k_l") && constraintGroup.find("k_l").isDouble()) {
                    pImpl->k_l = constraintGroup.find("k_l").asDouble();
                    yInfo() << "k_l: " << pImpl->k_l;
                }
            } // another option
            else {
                yError() << LogPrefix << "the parameter key is not defined: " << constraintKey;
                return false;
            }
        }
    }
    else {
        yInfo() << "CUSTOM CONSTRAINTS are not defined in xml file.";
    }

    // set base velocity constraint to zero if the base is fixed
    if (pImpl->useFixedBase) {
        pImpl->baseVelocityLowerLimit.resize(6);
        pImpl->baseVelocityLowerLimit.zero();
        pImpl->baseVelocityUpperLimit.resize(6);
        pImpl->baseVelocityUpperLimit.zero();

        yInfo() << "Using fixed base model, base velocity limits are set to zero";
    }

    // check sizes
    if (pImpl->custom_jointsVelocityLimitsNames.size()
        != pImpl->custom_jointsVelocityLimitsValues.size()) {
        yError() << "the joint velocity limits name and value size are not equal";
        return false;
    }
    if ((pImpl->customConstraintUpperBound.size() != pImpl->customConstraintLowerBound.size())
        && (pImpl->customConstraintLowerBound.size() != pImpl->customConstraintMatrix.rows())) {
        yError() << "the number of lower bound (" << pImpl->customConstraintLowerBound.size()
                 << "), upper buond(" << pImpl->customConstraintUpperBound.size()
                 << "), and cosntraint matrix rows(" << pImpl->customConstraintMatrix.rows()
                 << ") are not equal";

        return false;
    }
    if ((pImpl->customConstraintVariables.size() != pImpl->customConstraintMatrix.cols())) {
        yError() << "the number of constraint variables ("
                 << pImpl->customConstraintVariables.size() << "), and cosntraint matrix columns ("
                 << pImpl->customConstraintMatrix.cols() << ") are not equal";
        return false;
    }
    yInfo() << "******* DOF: " << modelLoader.model().getNrOfDOFs();
    for (size_t i = 0; i < pImpl->custom_jointsVelocityLimitsNames.size(); i++) {
        pImpl->custom_jointsVelocityLimitsIndexes.push_back(
            modelLoader.model().getJointIndex(pImpl->custom_jointsVelocityLimitsNames[i]));
        yInfo() << pImpl->custom_jointsVelocityLimitsNames[i] << " : "
                << pImpl->custom_jointsVelocityLimitsIndexes[i];
    }
    for (size_t i = 0; i < pImpl->customConstraintVariables.size(); i++) {
        pImpl->customConstraintVariablesIndex.push_back(
            modelLoader.model().getJointIndex(pImpl->customConstraintVariables[i]));
        yInfo() << pImpl->customConstraintVariables[i] << " : "
                << pImpl->customConstraintVariablesIndex[i];
    }

    // ==================================================================
    // CREATE LINK PAIR VECTOR FOR SECONDARY CALIBRATION AND PAIRWISED IK
    // ==================================================================

    if (!pImpl->createLinkPairs()) {
        askToStop();
        return false;
    }

    // ====================================
    // INITIALIZE INVERSE KINEMATICS SOLVER
    // ====================================

    if (pImpl->ikSolver == SolverIK::pairwised) {
        if (!pImpl->initializePairwisedInverseKinematicsSolver()) {
            askToStop();
            return false;
        }
    }
    else if (pImpl->ikSolver == SolverIK::global) {
        if (!pImpl->initializeGlobalInverseKinematicsSolver()) {
            askToStop();
            return false;
        }
    }
    else if (pImpl->ikSolver == SolverIK::integrationbased) {
        if (!pImpl->initializeIntegrationBasedInverseKinematicsSolver()) {
            askToStop();
            return false;
        }
    }

    // ==================================
    // INITIALIZE EXTERNAL BASE ESTIMATOR
    // ==================================

    if (pImpl->baseState == BaseState::external) {
        if (!pImpl->setupExternalBaseEstimator(config)) {
            return false;
        }
        if (!pImpl->setupExternalContactDetector(config)) {
            return false;
        }
    }

    // ===================
    // INITIALIZE RPC PORT
    // ===================

    std::string rpcPortName;
    if (!(config.check("rpcPortPrefix") && config.find("rpcPortPrefix").isString())) {
        rpcPortName = "/" + DeviceName + "/rpc:i";
    }
    else {
        rpcPortName = "/" + config.find("rpcPortPrefix").asString() + "/" + DeviceName + "/rpc:i";
    }

    if (!pImpl->rpcPort.open(rpcPortName)) {
        yError() << LogPrefix << "Unable to open rpc port " << rpcPortName;
        return false;
    }

    // Set rpc port reader
    pImpl->rpcPort.setReader(*pImpl->commandPro);

    return true;
}

bool HumanKinematicEstimator::impl::setupExternalBaseEstimator(yarp::os::Searchable& config)
{
    extLeggedOdom = std::make_unique<BipedalLocomotion::Estimators::LeggedOdometry>();

    auto originalHandler = std::make_shared<BipedalLocomotion::ParametersHandler::YarpImplementation>();
    originalHandler->set(config);
    BipedalLocomotion::ParametersHandler::IParametersHandler::shared_ptr parameterHandler = originalHandler;
    // set required parameter for base estimator
    parameterHandler->setParameter("sampling_period_in_s", period);

    if (!extLeggedOdom->initialize(parameterHandler, extKinDyn)) {
        yError() << LogPrefix << "Could not configure external base estimator - LeggedOdometry.";
        return false;
    }

    yInfo() << LogPrefix << "Configured external base estimator - LeggedOdometry - successfully.";
    return true;
}

bool HumanKinematicEstimator::impl::setupExternalContactDetector(yarp::os::Searchable& config)
{
    auto originalHandler = std::make_shared<BipedalLocomotion::ParametersHandler::YarpImplementation>();
    originalHandler->set(config);
    BipedalLocomotion::ParametersHandler::IParametersHandler::shared_ptr parameterHandler = originalHandler;
    bool ok{true};

    extSchmitt = std::make_unique<BipedalLocomotion::Contacts::SchmittTriggerDetector>();
    yarp::os::Bottle& schmittGroup = config.findGroup("SchmittTriggerParams");
    if (schmittGroup.isNull())
    {
        yError() << LogPrefix << "Could not configure external contact detector - schmitt trigger.";
        return false;
    }

    auto originalSchmittHandler = std::make_shared<BipedalLocomotion::ParametersHandler::YarpImplementation>();
    originalSchmittHandler->set(schmittGroup);
    BipedalLocomotion::ParametersHandler::IParametersHandler::shared_ptr parameterSchmittHandler = originalSchmittHandler;

    if (!extSchmitt->initialize(parameterSchmittHandler))
    {
        return false;
    }

    extSchmitt->resetState(leftFootString, true);
    extSchmitt->resetState(rightFootString, true);

    return true;
}

bool HumanKinematicEstimator::close()
{
    return true;
}

void HumanKinematicEstimator::run()
{

}


void HumanKinematicEstimator::impl::eraseSecondaryCalibration(const std::string& linkName)
{
    if (linkName == "") {
        secondaryCalibrationRotations.clear();
        secondaryCalibrationWorld = iDynTree::Transform::Identity();
        yInfo() << LogPrefix << "Discarding all the secondary calibration matrices";
    }
    else {
        if ((secondaryCalibrationRotations.find(linkName) != secondaryCalibrationRotations.end())) {
            secondaryCalibrationRotations.erase(linkName);
            yInfo() << LogPrefix << "Discarding the secondary calibration rotation matrix for link " << linkName;
        }
    }

}

void HumanKinematicEstimator::impl::selectChainJointsAndLinksForSecondaryCalibration(const std::string& linkName, const std::string& childLinkName,
                                              std::vector<iDynTree::JointIndex>& jointZeroIndices, std::vector<iDynTree::LinkIndex>& linkToCalibrateIndices)
{
    if (childLinkName == "") {
        // Select the chosen link [linkName] and the joints between the link and its parend link

        // add link to [linkToCalibrateIndices]
        linkToCalibrateIndices.push_back(kinDynComputations->model().getLinkIndex(linkName));

        // add joints to [jointZeroIndices]
        for (auto pairInfo = linkPairs.begin(); pairInfo != linkPairs.end(); pairInfo++)
        {
            if (pairInfo->parentFrameName == linkName) {
                for (size_t pairModelJointIndex = 0; pairModelJointIndex < pairInfo->pairModel.getNrOfJoints(); pairModelJointIndex++) {
                    std::string jointName = pairInfo->pairModel.getJointName(pairModelJointIndex);
                    jointZeroIndices.push_back(kinDynComputations->model().getJointIndex(jointName));
                }
                break;
            }
        }

    }
    else {
        // Create chain between the given parent link [linkName] and the child link [childLinkName] and select the joints and the links involved in the chain

        // create reduced model between parent and child frame
        iDynTree::Model chainModel;
        getReducedModel(kinDynComputations->model(), linkName, childLinkName, chainModel);

        // add links found in the submodel to [linkToCalibrateIndices]
        // TODO missing fake links that are frames
        for (size_t chainModelLinkIndex = 0; chainModelLinkIndex < chainModel.getNrOfLinks(); chainModelLinkIndex ++) {
            std::string chainLinkName = chainModel.getLinkName(chainModelLinkIndex);
            linkToCalibrateIndices.push_back(kinDynComputations->model().getLinkIndex(chainLinkName));
        }

        // add joints found in the submodel to [jointZeroIndices]
        for (size_t chainModelJointIndex = 0; chainModelJointIndex < chainModel.getNrOfJoints(); chainModelJointIndex++) {
            std::string jointName = chainModel.getJointName(chainModelJointIndex);
            jointZeroIndices.push_back(kinDynComputations->model().getJointIndex(jointName));
        }
    }
}

void HumanKinematicEstimator::impl::computeSecondaryCalibrationRotationsForChain(const std::vector<iDynTree::JointIndex>& jointZeroIndices, const iDynTree::Transform& refLinkForCalibrationTransform, const std::vector<iDynTree::LinkIndex>& linkToCalibrateIndices, const std::string& refLinkForCalibrationName)
{
    // initialize vectors
    iDynTree::VectorDynSize jointPos(jointConfigurationSolution);
    iDynTree::VectorDynSize jointVel(jointVelocitiesSolution);
    jointVel.zero();
    iDynTree::Twist baseVel;
    baseVel.zero();

    // setting to zero all the selected joints
    for (auto const& jointZeroIdx: jointZeroIndices) {
        jointPos.setVal(jointZeroIdx, 0);
    }

    kinDynComputations->setRobotState(linkTransformMatricesRaw.at(kinDynComputations->getFloatingBase()), jointPos, baseVel, jointVel, worldGravity);

    // computing the secondary calibration matrices
    for (auto const& linkToCalibrateIdx: linkToCalibrateIndices) {

        std::string linkToCalibrateName = kinDynComputations->model().getLinkName(linkToCalibrateIdx);
        if (!(wearableStorage.modelToWearable_LinkName.find(linkToCalibrateName) == wearableStorage.modelToWearable_LinkName.end())) {
            // discarding previous calibration
            eraseSecondaryCalibration(linkToCalibrateName);

            iDynTree::Transform linkTransformZero = kinDynComputations->getWorldTransform(linkToCalibrateName);

            // computing new calibration for orientation
            iDynTree::Rotation secondaryCalibrationRotation = linkTransformMatricesRaw.at(linkToCalibrateName).getRotation().inverse() * linkTransformZero.getRotation();

            // add new calibration
            secondaryCalibrationRotations.emplace(linkToCalibrateName,secondaryCalibrationRotation);
            yInfo() << LogPrefix << "secondary calibration for " << linkToCalibrateName << " is set";
        }
    }

    // computing the world calibration
    secondaryCalibrationWorld = iDynTree::Transform::Identity();
    if (refLinkForCalibrationName!="")
    {
        iDynTree::Transform linkForCalibrationTransform = kinDynComputations->getWorldTransform(refLinkForCalibrationName);
        secondaryCalibrationWorld = refLinkForCalibrationTransform * linkForCalibrationTransform.inverse();
        yInfo() << LogPrefix << "secondary calibration for the World is set";
    }
}

bool HumanKinematicEstimator::impl::updateExternalEstimatorAndDetector()
{
    if (!extSchmitt->advance()) {
        yError() << LogPrefix << "Could not update the Schmitt trigger detector" ;
    }


    if (!extLeggedOdom->setKinematics(iDynTree::toEigen(jointConfigurationSolution),
                                    iDynTree::toEigen(jointVelocitiesSolution)))
    {
        return false;
    }


    Eigen::Quaterniond qB = Eigen::Quaterniond(baseTransformSolution.getRotation().asQuaternion()(0),
                                               baseTransformSolution.getRotation().asQuaternion()(1),
                                               baseTransformSolution.getRotation().asQuaternion()(2),
                                               baseTransformSolution.getRotation().asQuaternion()(3));
    extLeggedOdom->resetEstimator(qB, iDynTree::toEigen(baseTransformSolution.getPosition()));

    if (itrCount < initThresh)
    {
        if (itrCount == initThresh-1)
        {
            if (!initializeEstimatorWorld())
            {
                yError() << LogPrefix << "Could not initialize estimator world." ;
                return false;
            }

            // need to advance the estimator to update the states
            if (!extLeggedOdom->advance()) {
                yError() << LogPrefix << "Could not advance the external base estimator after initialization." ;
                //return false;
            }

            // use the updated states to get the ground plane information
            // assuming that the link with fixed frame is in flat surface contact with the ground plane
            auto fixedFrameIdx = extLeggedOdom->getFixedFrameIdx();
            auto w_H_f0 = extLeggedOdom->modelComputations().kinDyn()->getWorldTransform(fixedFrameIdx);
            flatContactPlaneInclinationRoll =  w_H_f0.getRotation().asRPY()(0);
            flatContactPlaneInclinationPitch =  w_H_f0.getRotation().asRPY()(1);
            flatContactPlaneHeight = w_H_f0.getPosition()(2);
        }
        itrCount++;
        return true;
    }

    BipedalLocomotion::Contacts::EstimatedContactList contactMap;
    contactMap = extSchmitt->getOutput();
    for ( auto& [name, contact] : contactMap )
    {
        extLeggedOdom->setContactStatus(name, contact.isActive, contact.switchTime);

        if (name == leftFootString) {
            auto size = lfContact.size();
            lfContact.conservativeResize(size+1);
            lfTime.conservativeResize(size+1);
            lfContact(size) = static_cast<int>(contact.isActive);
            lfTime(size) = contact.lastUpdateTime;
            yInfo() << "LF Contact: " << static_cast<int>(contact.isActive);
        }

        if (name == rightFootString) {
            auto size = rfContact.size();
            rfContact.conservativeResize(size+1);
            rfTime.conservativeResize(size+1);
            rfContact(size) = static_cast<int>(contact.isActive);
            rfTime(size) = contact.lastUpdateTime;
            yInfo() << "RF Contact: " << static_cast<int>(contact.isActive);
        }
    }

    if (!extLeggedOdom->advance()) {
        yError() << LogPrefix << "Could not update the external base estimator" ;
        //return false;
    }

    // reset fixed frame pose using contact plane
    auto currentFixedFrame = extLeggedOdom->getFixedFrameIdx();

    auto worldFpose = extLeggedOdom->getFixedFramePose();
    Eigen::Vector3d worldFpos = worldFpose.translation();
    worldFpos(2) = flatContactPlaneHeight;
    worldFpose.translation(worldFpos);
    iDynTree::Rotation w_R_f;
    iDynTree::toEigen(w_R_f) = worldFpose.rotation();
    w_R_f = iDynTree::Rotation::RPY(flatContactPlaneInclinationRoll,
                                    flatContactPlaneInclinationPitch,
                                    w_R_f.asRPY()(2));

    Eigen::Quaterniond quatF = Eigen::Quaterniond(w_R_f.asQuaternion()(0),
                                                  w_R_f.asQuaternion()(1),
                                                  w_R_f.asQuaternion()(2),
                                                  w_R_f.asQuaternion()(3));

    extLeggedOdom->changeFixedFrame(currentFixedFrame,
                                    quatF,
                                    worldFpos);

    auto out = extLeggedOdom->getOutput();

    iDynTree::Vector3 linV, angV;
    iDynTree::toEigen(linV) = out.baseTwist.head<3>();
    iDynTree::toEigen(angV) = out.baseTwist.tail<3>();
//     baseVelocitySolution.setLinearVec3(linV);
//     baseVelocitySolution.setAngularVec3(angV);
    loTwist.setLinearVec3(linV);
    loTwist.setAngularVec3(angV);

    iDynTree::Transform estTransform;
    iDynTree::Matrix4x4 pose;
    iDynTree::toEigen(pose) = out.basePose.transform();
//     iDynTree::toEigen(pose).block<3, 3>(0, 0) = iDynTree::toEigen(linkTransformMatricesRaw.at("Pelvis").getRotation());
//     baseTransformSolution.fromHomogeneousTransform(pose);
    loPose.fromHomogeneousTransform(pose);
    estTransform.fromHomogeneousTransform(pose);

//     auto pEst = baseTransformSolution.getPosition();
//     auto rpyEst = baseTransformSolution.getRotation().asRPY();

    auto pEst = estTransform.getPosition();
    auto rpyEst = estTransform.getRotation().asRPY();
    auto estSize = estTime.rows();
    estTime.conservativeResize(estSize+1);
    extPos.conservativeResize(estSize+1, 3);
    extRot.conservativeResize(estSize+1, 3);
    extLinVel.conservativeResize(estSize+1, 3);
    extAngVel.conservativeResize(estSize+1, 3);
    extPos.row(estSize) << pEst(0), pEst(1), pEst(2);
    extRot.row(estSize) << rpyEst(0), rpyEst(1), rpyEst(2);
    extLinVel.row(estSize) << out.baseTwist(0), out.baseTwist(1), out.baseTwist(2);
    extAngVel.row(estSize) << out.baseTwist(3), out.baseTwist(4), out.baseTwist(5);

    auto w_H_b = linkTransformMatricesRaw.at("Pelvis");

    auto pOut = linkTransformMatricesRaw.at("Pelvis").getPosition();
    auto rpyOut = linkTransformMatricesRaw.at("Pelvis").getRotation().asRPY();
    auto baseTwist = linkVelocities.at("Pelvis");
    linkPos.conservativeResize(estSize+1, 3);
    linkRot.conservativeResize(estSize+1, 3);
    linkLinVel.conservativeResize(estSize+1, 3);
    linkAngVel.conservativeResize(estSize+1, 3);
    linkPos.row(estSize) << pOut(0), pOut(1), pOut(2);
    linkRot.row(estSize) << rpyOut(0), rpyOut(1), rpyOut(2);
    linkLinVel.row(estSize) << baseTwist(0), baseTwist(1), baseTwist(2);
    linkAngVel.row(estSize) << baseTwist(3), baseTwist(4), baseTwist(5);

    ikBasePos.conservativeResize(estSize+1, 3);
    ikBaseRot.conservativeResize(estSize+1, 3);
    ikBaseLinVel.conservativeResize(estSize+1, 3);
    ikBaseAngVel.conservativeResize(estSize+1, 3);
    ikBasePos.row(estSize) << iDynTree::toEigen(baseTransformSolution.getPosition());
    ikBaseRot.row(estSize) << iDynTree::toEigen(baseTransformSolution.getRotation().asRPY());
    ikBaseLinVel.row(estSize) << iDynTree::toEigen(baseVelocitySolution.getLinearVec3());
    ikBaseAngVel.row(estSize) << iDynTree::toEigen(baseVelocitySolution.getAngularVec3());
    outjointPos.conservativeResize(estSize+1, kinDynComputations->getNrOfDegreesOfFreedom());
    outjointVel.conservativeResize(estSize+1, kinDynComputations->getNrOfDegreesOfFreedom());
    outjointVelFilt.conservativeResize(estSize+1, kinDynComputations->getNrOfDegreesOfFreedom());
    for (int isx =0; isx < kinDynComputations->getNrOfDegreesOfFreedom(); isx++)
    {
        outjointPos(estSize, isx) = jointConfigurationSolution(isx);
        outjointVel(estSize, isx) = jointVelocitiesSolution(isx);
    }

    fixedFrame.conservativeResize(estSize+1);
    fixedFrame(estSize) = currentFixedFrame;
    return true;
}

bool HumanKinematicEstimator::impl::initializeEstimatorWorld()
{
    if (!m_extEstimatorInitialized)
    {
        auto b_H_w =  linkTransformMatricesRaw.at("Pelvis").inverse(); //baseTransformSolution

        auto quat = b_H_w.getRotation().asQuaternion();
        if (!extLeggedOdom->resetEstimator("Pelvis",
                                           Eigen::Quaterniond(quat(0), quat(1), quat(2), quat(3)),
                                           iDynTree::toEigen(b_H_w.getPosition())))
        {
            return false;
        }

        m_extEstimatorInitialized = true;
    }

    return true;
}

bool HumanKinematicEstimator::impl::logData()
{
    matioCpp::File file = matioCpp::File::Create("out-HDE-matiocpp.mat");
    matioCpp::MultiDimensionalArray<double> outEstPos{"estBasePos",
                                                      {static_cast<std::size_t>(extPos.rows()), static_cast<std::size_t>(extPos.cols())},
                                                      extPos.data()};
    matioCpp::MultiDimensionalArray<double> outEstRot{"estBaseRot",
                                                      {static_cast<std::size_t>(extRot.rows()), static_cast<std::size_t>(extRot.cols())},
                                                      extRot.data()};
    matioCpp::MultiDimensionalArray<double> outLinkPos{"linkBasePos",
                                                      {static_cast<std::size_t>(linkPos.rows()), static_cast<std::size_t>(linkPos.cols())},
                                                      linkPos.data()};
    matioCpp::MultiDimensionalArray<double> outLinkRot{"linkBaseRot",
                                                      {static_cast<std::size_t>(linkRot.rows()), static_cast<std::size_t>(linkRot.cols())},
                                                      linkRot.data()};
    matioCpp::MultiDimensionalArray<double> outEstLinVel{"estBaseLinVel",
                                                      {static_cast<std::size_t>(extLinVel.rows()), static_cast<std::size_t>(extLinVel.cols())},
                                                      extLinVel.data()};
    matioCpp::MultiDimensionalArray<double> outEstAngVel{"estBaseAngVel",
                                                      {static_cast<std::size_t>(extAngVel.rows()), static_cast<std::size_t>(extAngVel.cols())},
                                                      extAngVel.data()};
    matioCpp::MultiDimensionalArray<double> outLinkLinVel{"linkBaseLinVel",
                                                      {static_cast<std::size_t>(linkLinVel.rows()), static_cast<std::size_t>(linkLinVel.cols())},
                                                      linkLinVel.data()};
    matioCpp::MultiDimensionalArray<double> outLinkAngVel{"linkBaseAngVel",
                                                      {static_cast<std::size_t>(linkAngVel.rows()), static_cast<std::size_t>(linkAngVel.cols())},
                                                      linkAngVel.data()};

    matioCpp::MultiDimensionalArray<double> outBaseLinkPos{"baseTransformSolutionPos",
                                                      {static_cast<std::size_t>(ikBasePos.rows()), static_cast<std::size_t>(ikBasePos.cols())},
                                                      ikBasePos.data()};
    matioCpp::MultiDimensionalArray<double> outBaseLinkRot{"baseTransformSolutionRot",
                                                      {static_cast<std::size_t>(ikBaseRot.rows()), static_cast<std::size_t>(ikBaseRot.cols())},
                                                      ikBaseRot.data()};

    matioCpp::MultiDimensionalArray<double> outBaseLinkLinVel{"baseVelocitySolutionLinVel",
                                                      {static_cast<std::size_t>(ikBaseLinVel.rows()), static_cast<std::size_t>(ikBaseLinVel.cols())},
                                                      ikBaseLinVel.data()};
    matioCpp::MultiDimensionalArray<double> outBaseLinkAngVel{"baseVelocitySolutionAngVel",
                                                      {static_cast<std::size_t>(ikBaseAngVel.rows()), static_cast<std::size_t>(ikBaseAngVel.cols())},
                                                      ikBaseAngVel.data()};

    matioCpp::MultiDimensionalArray<double> outJPos{"jPos",
                                                      {static_cast<std::size_t>(outjointPos.rows()), static_cast<std::size_t>(outjointPos.cols())},
                                                      outjointPos.data()};
    matioCpp::MultiDimensionalArray<double> outJVel{"jVel",
                                                      {static_cast<std::size_t>(outjointVel.rows()), static_cast<std::size_t>(outjointVel.cols())},
                                                      outjointVel.data()};
    matioCpp::MultiDimensionalArray<double> outJVelFilt{"jVelFilt",
                                                      {static_cast<std::size_t>(outjointVelFilt.rows()), static_cast<std::size_t>(outjointVelFilt.cols())},
                                                      outjointVelFilt.data()};

    matioCpp::MultiDimensionalArray<double> outlfWrench{"lfWrench",
                                                      {static_cast<std::size_t>(lfWrench.rows()), static_cast<std::size_t>(lfWrench.cols())},
                                                      lfWrench.data()};
    matioCpp::MultiDimensionalArray<double> outrfWrench{"rfWrench",
                                                      {static_cast<std::size_t>(rfWrench.rows()), static_cast<std::size_t>(rfWrench.cols())},
                                                      rfWrench.data()};

    auto outContactlf = BipedalLocomotion::Conversions::tomatioCpp(lfContact, "estLFContact");
    auto outContactrf = BipedalLocomotion::Conversions::tomatioCpp(rfContact, "estRFContact");
    auto outlfForceZ = BipedalLocomotion::Conversions::tomatioCpp(lfForce, "LFForceZ");
    auto outrfForceZ = BipedalLocomotion::Conversions::tomatioCpp(rfForce, "RFForceZ");
    auto outContactlfTime = BipedalLocomotion::Conversions::tomatioCpp(lfTime, "estLFContactTime");
    auto outContactrfTime = BipedalLocomotion::Conversions::tomatioCpp(rfTime, "estRFContactTime");


    auto outBaseTime = BipedalLocomotion::Conversions::tomatioCpp(estTime, "estBaseTime");
    auto outFixedFrame = BipedalLocomotion::Conversions::tomatioCpp(fixedFrame, "estFixedFrame");


    bool write_ok{true};

    write_ok = write_ok && file.write(outEstPos);
    write_ok = write_ok && file.write(outEstRot);
    write_ok = write_ok && file.write(outLinkPos);
    write_ok = write_ok && file.write(outLinkRot);
    write_ok = write_ok && file.write(outEstLinVel);
    write_ok = write_ok && file.write(outEstAngVel);
    write_ok = write_ok && file.write(outLinkLinVel);
    write_ok = write_ok && file.write(outLinkAngVel);

    write_ok = write_ok && file.write(outlfForceZ);
    write_ok = write_ok && file.write(outrfForceZ);
    write_ok = write_ok && file.write(outContactrf);
    write_ok = write_ok && file.write(outContactlf);
    write_ok = write_ok && file.write(outContactlfTime);
    write_ok = write_ok && file.write(outContactrfTime);

    write_ok = write_ok && file.write(outJPos);
    write_ok = write_ok && file.write(outJVel);
    write_ok = write_ok && file.write(outJVelFilt);
    write_ok = write_ok && file.write(outFixedFrame);

    write_ok = write_ok && file.write(outBaseLinkPos);
    write_ok = write_ok && file.write(outBaseLinkRot);
    write_ok = write_ok && file.write(outBaseLinkLinVel);
    write_ok = write_ok && file.write(outBaseLinkAngVel);

    if (!write_ok)
    {
        yError() << LogPrefix << "Could not write to file." ;
        return false;
    }

    return true;
}

bool HumanKinematicEstimator::impl::applyRpcCommand()
{
    // check is the choosen links are valid
    std::string linkName = commandPro->parentLinkName;
    std::string childLinkName = commandPro->childLinkName;
    if (!(linkName == "") && (wearableStorage.modelToWearable_LinkName.find(linkName) == wearableStorage.modelToWearable_LinkName.end())) {
        yWarning() << LogPrefix << "link " << linkName << " choosen for secondaty calibration is not valid";
        return false;
    }
    if (!(childLinkName == "") && (wearableStorage.modelToWearable_LinkName.find(childLinkName) == wearableStorage.modelToWearable_LinkName.end())) {
        yWarning() << LogPrefix << "link " << childLinkName << " choosen for secondaty calibration is not valid";
        return false;
    }

    // initialize buffer variable for calibration
    std::vector<iDynTree::JointIndex> jointZeroIndices;
    std::vector<iDynTree::LinkIndex> linkToCalibrateIndices;
    iDynTree::Rotation secondaryCalibrationRotation;

    switch(commandPro->cmdStatus) {
    case rpcCommand::resetCalibration: {
        eraseSecondaryCalibration(linkName);
        break;
    }
    case rpcCommand::calibrateAll: {
        // Select all the links and the joints
        // add all the links of the model to [linkToCalibrateIndices]
        linkToCalibrateIndices.resize(kinDynComputations->getNrOfLinks());
        std::iota(linkToCalibrateIndices.begin(), linkToCalibrateIndices.end(), 0);

        // add all the joints of the model to [jointZeroIndices]
        jointZeroIndices.resize(kinDynComputations->getNrOfDegreesOfFreedom());
        std::iota(jointZeroIndices.begin(), jointZeroIndices.end(), 0);

        // Compute secondary calibration for the selected links setting to zero the given joints
        computeSecondaryCalibrationRotationsForChain(jointZeroIndices, iDynTree::Transform::Identity(), linkToCalibrateIndices, "");
        break;
    }
    case rpcCommand::calibrateAllWithWorld: {
        // Check if the chose baseLink exist in the model
        std::string refLinkForCalibrationName = commandPro->refLinkName;
        if(!(kinDynComputations->getRobotModel().isFrameNameUsed(refLinkForCalibrationName)))
        {
            yWarning() << LogPrefix << "link " << refLinkForCalibrationName << " choosen as base for secondaty calibration is not valid";
            return false;
        }

        // Select all the links and the joints
        // add all the links of the model to [linkToCalibrateIndices]
        linkToCalibrateIndices.resize(kinDynComputations->getNrOfLinks());
        std::iota(linkToCalibrateIndices.begin(), linkToCalibrateIndices.end(), 0);

        // add all the joints of the model to [jointZeroIndices]
        jointZeroIndices.resize(kinDynComputations->getNrOfDegreesOfFreedom());
        std::iota(jointZeroIndices.begin(), jointZeroIndices.end(), 0);

        // Compute secondary calibration for the selected links setting to zero the given joints
        computeSecondaryCalibrationRotationsForChain(jointZeroIndices, iDynTree::Transform::Identity(), linkToCalibrateIndices, refLinkForCalibrationName);
        break;
    }
    case rpcCommand::calibrate: {
        // Select the joints to be set to zero and the link to be add the secondary calibration
        selectChainJointsAndLinksForSecondaryCalibration(linkName, childLinkName, jointZeroIndices, linkToCalibrateIndices);
        // Compute secondary calibration for the selected links setting to zero the given joints
        computeSecondaryCalibrationRotationsForChain(jointZeroIndices, iDynTree::Transform::Identity(), linkToCalibrateIndices, "");
        break;
    }
    case rpcCommand::calibrateSubTree: {
        // Select the joints to be set to zero and the link to be add the secondary calibration
        selectChainJointsAndLinksForSecondaryCalibration(linkName, childLinkName, jointZeroIndices, linkToCalibrateIndices);
        // Compute secondary calibration for the selected links setting to zero the given joints
        computeSecondaryCalibrationRotationsForChain(jointZeroIndices, iDynTree::Transform::Identity(), linkToCalibrateIndices, "");
    }
    case rpcCommand::calibrateRelativeLink: {
        eraseSecondaryCalibration(childLinkName);
        // Compute the relative transform at zero configuration
        // setting to zero all the joints
        iDynTree::VectorDynSize jointPos;
        jointPos.resize(jointConfigurationSolution.size());
        jointPos.zero();
        kinDynComputations->setJointPos(jointPos);
        iDynTree::Rotation relativeRotationZero = kinDynComputations->getWorldTransform(linkName).getRotation().inverse() * kinDynComputations->getWorldTransform(childLinkName).getRotation();
        secondaryCalibrationRotation = linkTransformMatricesRaw.at(childLinkName).getRotation().inverse() * linkTransformMatrices.at(linkName).getRotation() * relativeRotationZero;
        secondaryCalibrationRotations.emplace(childLinkName,secondaryCalibrationRotation);
        yInfo() << LogPrefix << "secondary calibration for " << childLinkName << " is set";
        break;
     }
    case rpcCommand::setRotationOffset: {
        eraseSecondaryCalibration(linkName);
        secondaryCalibrationRotation = iDynTree::Rotation::RPY( 3.14 * commandPro->roll / 180 , 3.14 * commandPro->pitch / 180 , 3.14 * commandPro->yaw / 180 );
        // add new calibration
        secondaryCalibrationRotations.emplace(linkName,secondaryCalibrationRotation);
        yInfo() << LogPrefix << "secondary calibration for " << linkName << " is set";
        break;
    }
    default: {
        yWarning() << LogPrefix << "Command not valid";
        return false;
    }
    }

    return true;
}

bool HumanKinematicEstimator::impl::getLinkTransformFromInputData(
    std::unordered_map<std::string, iDynTree::Transform>& transforms)
{
    for (const auto& linkMapEntry : wearableStorage.modelToWearable_LinkName) {
        const ModelLinkName& modelLinkName = linkMapEntry.first;
        const WearableLinkName& wearableLinkName = linkMapEntry.second;

        if (wearableStorage.linkSensorsMap.find(wearableLinkName)
                == wearableStorage.linkSensorsMap.end()
            || !wearableStorage.linkSensorsMap.at(wearableLinkName)) {
            yError() << LogPrefix << "Failed to get" << wearableLinkName
                     << "sensor from the device. Something happened after configuring it.";
            return false;
        }

        const wearable::SensorPtr<const sensor::IVirtualLinkKinSensor> sensor =
            wearableStorage.linkSensorsMap.at(wearableLinkName);

        if (!sensor) {
            yError() << LogPrefix << "Sensor" << wearableLinkName
                     << "has been added but not properly configured";
            return false;
        }

        if (sensor->getSensorStatus() != sensor::SensorStatus::Ok) {
            yError() << LogPrefix << "The sensor status of" << sensor->getSensorName()
                     << "is not ok (" << static_cast<double>(sensor->getSensorStatus()) << ")";
            return false;
        }

        wearable::Vector3 position;
        if (!sensor->getLinkPosition(position)) {
            yError() << LogPrefix << "Failed to read link position from virtual link sensor";
            return false;
        }

        iDynTree::Position pos(position.at(0), position.at(1), position.at(2));

        wearable::Quaternion orientation;
        if (!sensor->getLinkOrientation(orientation)) {
            yError() << LogPrefix << "Failed to read link orientation from virtual link sensor";
            return false;
        }

        iDynTree::Rotation rotation;
        rotation.fromQuaternion({orientation.data(), 4});

        iDynTree::Transform transform(rotation, pos);

        // Note that this map is used during the IK step for setting a target transform to a
        // link of the model. For this reason the map keys are model names.
        transforms[modelLinkName] = std::move(transform);
    }

    return true;
}

bool HumanKinematicEstimator::impl::applySecondaryCalibration(
        const std::unordered_map<std::string, iDynTree::Transform> &transforms_in, std::unordered_map<std::string, iDynTree::Transform> &transforms_out)
{
    transforms_out = transforms_in;
    for (const auto& linkMapEntry : wearableStorage.modelToWearable_LinkName) {
        const ModelLinkName& modelLinkName = linkMapEntry.first;

        // Apply secondary calibration for rotation
        auto secondaryCalibrationRotationsIt = secondaryCalibrationRotations.find(modelLinkName);
        if (!(secondaryCalibrationRotationsIt
              == secondaryCalibrationRotations.end())) {

            iDynTree::Transform calibrationTransform;
            calibrationTransform.setPosition(iDynTree::Position(0,0,0));
            calibrationTransform.setRotation(secondaryCalibrationRotationsIt->second);

            transforms_out[modelLinkName] = transforms_out[modelLinkName] * calibrationTransform;
        }

        transforms_out[modelLinkName] = secondaryCalibrationWorld * transforms_out[modelLinkName];
    }

    return true;
}

bool HumanKinematicEstimator::impl::getLinkVelocityFromInputData(
    std::unordered_map<std::string, iDynTree::Twist>& velocities)
{
    for (const auto& linkMapEntry : wearableStorage.modelToWearable_LinkName) {
        const ModelLinkName& modelLinkName = linkMapEntry.first;
        const WearableLinkName& wearableLinkName = linkMapEntry.second;

        if (wearableStorage.linkSensorsMap.find(wearableLinkName)
                == wearableStorage.linkSensorsMap.end()
            || !wearableStorage.linkSensorsMap.at(wearableLinkName)) {
            yError() << LogPrefix << "Failed to get" << wearableLinkName
                     << "sensor from the device. Something happened after configuring it.";
            return false;
        }

        const wearable::SensorPtr<const sensor::IVirtualLinkKinSensor> sensor =
            wearableStorage.linkSensorsMap.at(wearableLinkName);

        if (!sensor) {
            yError() << LogPrefix << "Sensor" << wearableLinkName
                     << "has been added but not properly configured";
            return false;
        }

        if (sensor->getSensorStatus() != sensor::SensorStatus::Ok) {
            yError() << LogPrefix << "The sensor status of" << sensor->getSensorName()
                     << "is not ok (" << static_cast<double>(sensor->getSensorStatus()) << ")";
            return false;
        }

        wearable::Vector3 linearVelocity;
        if (!sensor->getLinkLinearVelocity(linearVelocity)) {
            yError() << LogPrefix << "Failed to read link linear velocity from virtual link sensor";
            return false;
        }

        wearable::Vector3 angularVelocity;
        if (!sensor->getLinkAngularVelocity(angularVelocity)) {
            yError() << LogPrefix
                     << "Failed to read link angular velocity from virtual link sensor";
            return false;
        }

        velocities[modelLinkName].setVal(0, linearVelocity.at(0));
        velocities[modelLinkName].setVal(1, linearVelocity.at(1));
        velocities[modelLinkName].setVal(2, linearVelocity.at(2));
        velocities[modelLinkName].setVal(3, angularVelocity.at(0));
        velocities[modelLinkName].setVal(4, angularVelocity.at(1));
        velocities[modelLinkName].setVal(5, angularVelocity.at(2));
    }

    return true;
}

bool HumanKinematicEstimator::impl::getJointAnglesFromInputData(iDynTree::VectorDynSize& jointAngles)
{
    for (const auto& jointMapEntry : wearableStorage.modelToWearable_JointInfo) {
        const ModelJointName& modelJointName = jointMapEntry.first;
        const WearableJointInfo& wearableJointInfo = jointMapEntry.second;

        if (wearableStorage.jointSensorsMap.find(wearableJointInfo.name)
                == wearableStorage.jointSensorsMap.end()
            || !wearableStorage.jointSensorsMap.at(wearableJointInfo.name)) {
            yError() << LogPrefix << "Failed to get" << wearableJointInfo.name
                     << "sensor from the device. Something happened after configuring it.";
            return false;
        }

        const wearable::SensorPtr<const sensor::IVirtualSphericalJointKinSensor> sensor =
            wearableStorage.jointSensorsMap.at(wearableJointInfo.name);

        if (!sensor) {
            yError() << LogPrefix << "Sensor" << wearableJointInfo.name
                     << "has been added but not properly configured";
            return false;
        }

        if (sensor->getSensorStatus() != sensor::SensorStatus::Ok) {
            yError() << LogPrefix << "The sensor status of " << sensor->getSensorName()
                     << " is not ok (" << static_cast<double>(sensor->getSensorStatus()) << ")";
            return false;
        }

        Vector3 anglesXYZ;
        if (!sensor->getJointAnglesAsRPY(anglesXYZ)) {
            yError() << LogPrefix << "Failed to read joint angles from virtual joint sensor";
            return false;
        }

        // Since anglesXYZ describes a spherical joint, take the right component
        // (specified in the configuration file)
        // TODO: we still need to validate the Xsens convention. Particularly, the zeros of
        //       the joint angles might be different.
        jointAngles.setVal(humanModel.getJointIndex(modelJointName),
                           anglesXYZ[wearableJointInfo.index]);
    }

    return true;
}

bool HumanKinematicEstimator::impl::createLinkPairs()
{
    // Get the model link names according to the modelToWearable link sensor map
    const size_t nrOfSegments = wearableStorage.modelToWearable_LinkName.size();

    segments.resize(nrOfSegments);

    unsigned segmentIndex = 0;
    for (size_t linkIndex = 0; linkIndex < humanModel.getNrOfLinks(); ++linkIndex) {
        // Get the name of the link from the model and its prefix from iWear
        std::string modelLinkName = humanModel.getLinkName(linkIndex);

        if (wearableStorage.modelToWearable_LinkName.find(modelLinkName)
            == wearableStorage.modelToWearable_LinkName.end()) {
            continue;
        }

        // TODO check if we need this initialization
        // segments[segmentIndex].velocities.zero();

        // Store the name of the link as segment name
        segments[segmentIndex].segmentName = modelLinkName;
        segmentIndex++;
    }

    // Get all the possible pairs composing the model
    std::vector<std::pair<std::string, std::string>> pairNames;
    std::vector<std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex>> pairSegmentIndeces;

    // Get the link pair names
    createEndEffectorsPairs(humanModel, segments, pairNames, pairSegmentIndeces);
    linkPairs.reserve(pairNames.size());

    for (unsigned index = 0; index < pairNames.size(); ++index) {
        LinkPairInfo pairInfo;

        pairInfo.parentFrameName = pairNames[index].first;
        pairInfo.parentFrameSegmentsIndex = pairSegmentIndeces[index].first;

        pairInfo.childFrameName = pairNames[index].second;
        pairInfo.childFrameSegmentsIndex = pairSegmentIndeces[index].second;

        yInfo() << "getting the reduced model from: " << pairInfo.parentFrameName << " to " << pairInfo.childFrameName;
        // Get the reduced pair model
        if (!getReducedModel(humanModel,
                             pairInfo.parentFrameName,
                             pairInfo.childFrameName,
                             pairInfo.pairModel)) {

            yWarning() << LogPrefix << "failed to get reduced model for the segment pair "
                       << pairInfo.parentFrameName.c_str() << ", "
                       << pairInfo.childFrameName.c_str();
            continue;
        }

        // Move the link pair instance into the vector
        linkPairs.push_back(std::move(pairInfo));
    }

    return true;
}

bool HumanKinematicEstimator::impl::initializePairwisedInverseKinematicsSolver()
{
    for (auto pairInfo = linkPairs.begin(); pairInfo != linkPairs.end(); pairInfo++)
    {
        // Allocate the ik solver
        pairInfo->ikSolver = std::make_unique<iDynTree::InverseKinematics>();

        // Set ik parameters
        pairInfo->ikSolver->setVerbosity(1);
        pairInfo->ikSolver->setLinearSolverName(linearSolverName);
        pairInfo->ikSolver->setMaxIterations(maxIterationsIK);
        pairInfo->ikSolver->setCostTolerance(costTolerance);
        pairInfo->ikSolver->setDefaultTargetResolutionMode(
            iDynTree::InverseKinematicsTreatTargetAsConstraintNone);
        pairInfo->ikSolver->setRotationParametrization(
            iDynTree::InverseKinematicsRotationParametrizationRollPitchYaw);

        // Set ik model
        if (!pairInfo->ikSolver->setModel(pairInfo->pairModel)) {
            yWarning() << LogPrefix << "failed to configure IK solver for the segment pair"
                       << pairInfo->parentFrameName.c_str() << ", "
                       << pairInfo->childFrameName.c_str() << " Skipping pair";
            continue;
        }

        // Add parent link as fixed base constraint with identity transform
        pairInfo->ikSolver->addFrameConstraint(pairInfo->parentFrameName,
                                              iDynTree::Transform::Identity());

        // Add child link as a target and set initial transform to be identity
        pairInfo->ikSolver->addTarget(pairInfo->childFrameName, iDynTree::Transform::Identity());

        // Add target position and rotation weights
        pairInfo->positionTargetWeight = posTargetWeight;
        pairInfo->rotationTargetWeight = rotTargetWeight;

        // Add cost regularization term
        pairInfo->costRegularization = costRegularization;

        // Get floating base for the pair model
        pairInfo->floatingBaseIndex = pairInfo->pairModel.getFrameLink(
            pairInfo->pairModel.getFrameIndex(pairInfo->parentFrameName));

        // Set ik floating base
        if (!pairInfo->ikSolver->setFloatingBaseOnFrameNamed(
                pairInfo->pairModel.getLinkName(pairInfo->floatingBaseIndex))) {
            yError() << "Failed to set floating base frame for the segment pair"
                     << pairInfo->parentFrameName.c_str() << ", " << pairInfo->childFrameName.c_str()
                     << " Skipping pair";
            return false;
        }

        // Set initial joint positions size
        pairInfo->sInitial.resize(pairInfo->pairModel.getNrOfJoints());

        // Obtain the joint location index in full model and the lenght of DoFs i.e joints map
        // This information will be used to put the IK solutions together for the full model
        std::vector<std::string> solverJoints;

        // Resize to number of joints in the pair model
        solverJoints.resize(pairInfo->pairModel.getNrOfJoints());

        for (int i = 0; i < pairInfo->pairModel.getNrOfJoints(); i++) {
            solverJoints[i] = pairInfo->pairModel.getJointName(i);
        }

        pairInfo->consideredJointLocations.reserve(solverJoints.size());
        for (auto& jointName : solverJoints) {
            iDynTree::JointIndex jointIndex = humanModel.getJointIndex(jointName);
            if (jointIndex == iDynTree::JOINT_INVALID_INDEX) {
                yWarning() << LogPrefix << "IK considered joint " << jointName
                           << " not found in the complete model";
                continue;
            }
            iDynTree::IJointConstPtr joint = humanModel.getJoint(jointIndex);

            // Save location index and length of each DoFs
            pairInfo->consideredJointLocations.push_back(
                std::pair<size_t, size_t>(joint->getDOFsOffset(), joint->getNrOfDOFs()));
        }

        // Set the joint configurations size and initialize to zero
        pairInfo->jointConfigurations.resize(solverJoints.size());
        pairInfo->jointConfigurations.zero();

        // Set the joint velocities size and initialize to zero
        pairInfo->jointVelocities.resize(solverJoints.size());
        pairInfo->jointVelocities.zero();

        // Save the indeces
        // TODO: check if link or frame
        pairInfo->parentFrameModelIndex = pairInfo->pairModel.getFrameIndex(pairInfo->parentFrameName);
        pairInfo->childFrameModelIndex = pairInfo->pairModel.getFrameIndex(pairInfo->childFrameName);

        // Configure KinDynComputation
        pairInfo->kinDynComputations =
            std::unique_ptr<iDynTree::KinDynComputations>(new iDynTree::KinDynComputations());
        pairInfo->kinDynComputations->loadRobotModel(pairInfo->pairModel);

        // Configure relative Jacobian
        pairInfo->relativeJacobian.resize(6, pairInfo->pairModel.getNrOfDOFs());
        pairInfo->relativeJacobian.zero();
    }

    // Initialize IK Worker Pool
    ikPool = std::unique_ptr<IKWorkerPool>(new IKWorkerPool(ikPoolSize, linkPairs, segments));
    if (!ikPool) {
        yError() << LogPrefix << "failed to create IK worker pool";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);

        // Set the initial solution in the middle between lower and upper limits
        for (auto& linkPair : linkPairs) {
            for (size_t i = 0; i < linkPair.pairModel.getNrOfJoints(); i++) {
                double minJointLimit = linkPair.pairModel.getJoint(i)->getMinPosLimit(i);
                double maxJointLimit = linkPair.pairModel.getJoint(i)->getMaxPosLimit(i);
                double averageJointLimit = (minJointLimit + maxJointLimit) / 2.0;
                linkPair.sInitial.setVal(i, averageJointLimit);
            }
        }
    }

    return true;
}

bool HumanKinematicEstimator::impl::initializeGlobalInverseKinematicsSolver()
{
    // Set global ik parameters
    globalIK.setVerbosity(1);
    globalIK.setLinearSolverName(linearSolverName);
    globalIK.setMaxIterations(maxIterationsIK);
    globalIK.setCostTolerance(costTolerance);
    globalIK.setDefaultTargetResolutionMode(iDynTree::InverseKinematicsTreatTargetAsConstraintNone);
    globalIK.setRotationParametrization(
        iDynTree::InverseKinematicsRotationParametrizationRollPitchYaw);

    if (!globalIK.setModel(humanModel)) {
        yError() << LogPrefix << "globalIK: failed to load the model";
        return false;
    }

    if (!globalIK.setFloatingBaseOnFrameNamed(floatingBaseFrame)) {
        yError() << LogPrefix << "Failed to set the globalIK floating base frame on link"
                 << floatingBaseFrame;
        return false;
    }

    if (!addInverseKinematicTargets()) {
        yError() << LogPrefix << "Failed to set the globalIK targets";
        return false;
    }

    // Set global Inverse Velocity Kinematics parameters
    inverseVelocityKinematics.setResolutionMode(inverseVelocityKinematicsSolver);
    inverseVelocityKinematics.setRegularization(costRegularization);

    if (!inverseVelocityKinematics.setModel(humanModel)) {
        yError() << LogPrefix << "IBIK: failed to load the model";
        return false;
    }

    if (!inverseVelocityKinematics.setFloatingBaseOnFrameNamed(floatingBaseFrame)) {
        yError() << LogPrefix << "Failed to set the IBIK floating base frame on link"
                 << floatingBaseFrame;
        return false;
    }

    if (!addInverseVelocityKinematicsTargets()) {
        yError() << LogPrefix << "Failed to set the inverse velocity kinematics targets";
        return false;
    }
    return true;
}

bool HumanKinematicEstimator::impl::initializeIntegrationBasedInverseKinematicsSolver()
{
    // Initialize state integrator
    stateIntegrator.setInterpolatorType(iDynTreeHelper::State::integrator::trapezoidal);
    stateIntegrator.setNJoints(humanModel.getNrOfDOFs());

    iDynTree::VectorDynSize jointLowerLimits;
    jointLowerLimits.resize(humanModel.getNrOfDOFs());
    iDynTree::VectorDynSize jointUpperLimits;
    jointUpperLimits.resize(humanModel.getNrOfDOFs());
    for (int jointIndex = 0; jointIndex < humanModel.getNrOfDOFs(); jointIndex++) {
        jointLowerLimits.setVal(jointIndex, humanModel.getJoint(jointIndex)->getMinPosLimit(0));
        jointUpperLimits.setVal(jointIndex, humanModel.getJoint(jointIndex)->getMaxPosLimit(0));
    }
    stateIntegrator.setJointLimits(jointLowerLimits, jointUpperLimits);

    integralOrientationError.zero();

    // Set global Inverse Velocity Kinematics parameters
    inverseVelocityKinematics.setResolutionMode(inverseVelocityKinematicsSolver);
    // Set Regularization Term:
    inverseVelocityKinematics.setRegularization(costRegularization);

    if (!inverseVelocityKinematics.setModel(humanModel)) {
        yError() << LogPrefix << "IBIK: failed to load the model";
        return false;
    }

    if (!inverseVelocityKinematics.setFloatingBaseOnFrameNamed(floatingBaseFrame)) {
        yError() << LogPrefix << "Failed to set the IBIK floating base frame on link"
                 << floatingBaseFrame;
        return false;
    }

    if (!addInverseVelocityKinematicsTargets()) {
        yError() << LogPrefix << "Failed to set the inverse velocity kinematics targets";
        return false;
    }

    // =========================
    // SET CONSTRAINTS FOR IB-IK
    // =========================
    if (custom_jointsVelocityLimitsNames.size() != 0) {
        inverseVelocityKinematics.setCustomJointsVelocityLimit(custom_jointsVelocityLimitsIndexes,
                                                               custom_jointsVelocityLimitsValues);
    }
    if (baseVelocityUpperLimit.size() != 0) {
        inverseVelocityKinematics.setCustomBaseVelocityLimit(baseVelocityLowerLimit,
                                                             baseVelocityUpperLimit);
    }
    if (customConstraintVariablesIndex.size() != 0) {
        inverseVelocityKinematics.setCustomConstraintsJointsValues(customConstraintVariablesIndex,
                                                                   customConstraintUpperBound,
                                                                   customConstraintLowerBound,
                                                                   customConstraintMatrix,
                                                                   k_u,
                                                                   k_l);
    }

    inverseVelocityKinematics.setGeneralJointVelocityConstraints(
        integrationBasedJointVelocityLimit);

    inverseVelocityKinematics.setGeneralJointsUpperLowerConstraints(jointUpperLimits,
                                                                    jointLowerLimits);

    return true;
}

bool HumanKinematicEstimator::impl::solvePairwisedInverseKinematicsSolver()
{
    {
        std::lock_guard<std::mutex> lock(mutex);

        // Set link segments transformation and velocity
        for (size_t segmentIndex = 0; segmentIndex < segments.size(); segmentIndex++) {

            SegmentInfo& segmentInfo = segments.at(segmentIndex);
            segmentInfo.poseWRTWorld = linkTransformMatrices.at(segmentInfo.segmentName);
            segmentInfo.velocities = linkVelocities.at(segmentInfo.segmentName);
        }
    }

    // Call IK worker pool to solve
    ikPool->runAndWait();

    // Joint link pair ik solutions using joints map from link pairs initialization
    // to solution struct for exposing data through interface
    for (auto& linkPair : linkPairs) {
        size_t jointIndex = 0;
        for (auto& pairJoint : linkPair.consideredJointLocations) {

            // Check if it is a valid 1 DoF joint
            if (pairJoint.second == 1) {
                jointConfigurationSolution.setVal(pairJoint.first,
                                                  linkPair.jointConfigurations.getVal(jointIndex));
                jointVelocitiesSolution.setVal(pairJoint.first,
                                               linkPair.jointVelocities.getVal(jointIndex));

                linkPair.sInitial.setVal(jointIndex,
                                         linkPair.jointConfigurations.getVal(jointIndex));
                jointIndex++;
            }
            else {
                yWarning()
                    << LogPrefix
                    << " Invalid DoFs for the joint, skipping the ik solution for this joint";
                continue;
            }
        }
    }

    return true;
}

bool HumanKinematicEstimator::impl::solveGlobalInverseKinematicsSolver()
{
    // Set global IK initial condition
    if (!globalIK.setFullJointsInitialCondition(&baseTransformSolution,
                                                &jointConfigurationSolution)) {
        yError() << LogPrefix
                 << "Failed to set the joint configuration for initializing the global IK";
        return false;
    }

    // Update ik targets based on wearable input data
    if (!updateInverseKinematicTargets()) {
        yError() << LogPrefix << "Failed to update the targets for the global IK";
        return false;
    }

    // Use a postural task for regularization
    iDynTree::VectorDynSize posturalTaskJointAngles;
    posturalTaskJointAngles.resize(jointConfigurationSolution.size());
    posturalTaskJointAngles.zero();
    if (!globalIK.setDesiredFullJointsConfiguration(posturalTaskJointAngles, costRegularization)) {
        yError() << LogPrefix << "Failed to set the postural configuration of the IK";
        return false;
    }

    if (!globalIK.solve()) {
        yError() << LogPrefix << "Failed to solve global IK";
        return false;
    }

    // Get the global inverse kinematics solution
    globalIK.getFullJointsSolution(baseTransformSolution, jointConfigurationSolution);

    // INVERSE VELOCITY KINEMATICS
    // Set joint configuration
    if (!inverseVelocityKinematics.setConfiguration(baseTransformSolution,
                                                    jointConfigurationSolution)) {
        yError() << LogPrefix
                 << "Failed to set the joint configuration for initializing the inverse velocity "
                    "kinematics";
        return false;
    }

    // Update ivk velocity targets based on wearable input data
    if (!updateInverseVelocityKinematicTargets()) {
        yError() << LogPrefix << "Failed to update the targets for the inverse velocity kinematics";
        return false;
    }

    if (!inverseVelocityKinematics.solve()) {
        yError() << LogPrefix << "Failed to solve inverse velocity kinematics";
        return false;
    }

    inverseVelocityKinematics.getVelocitySolution(baseVelocitySolution, jointVelocitiesSolution);

    return true;
}

bool HumanKinematicEstimator::impl::solveIntegrationBasedInverseKinematics()
{
    // compute timestep
    double dt;
    if (lastTime < 0.0) {
        dt = period;
    }
    else {
        dt = yarp::os::Time::now() - lastTime;
    };
    lastTime = yarp::os::Time::now();

    // LINK VELOCITY CORRECTION
    iDynTree::KinDynComputations* computations = kinDynComputations.get();

    if (useDirectBaseMeasurement) {
//         computations->setRobotState(jointConfigurationSolution,
//                                     jointVelocitiesSolution,
//                                     worldGravity);
        computations->setRobotState(baseTransformSolution,
                                    jointConfigurationSolution,
                                    baseVelocitySolution,
                                    jointVelocitiesSolution,
                                    worldGravity);
    }
    else {
        computations->setRobotState(baseTransformSolution,
                                    jointConfigurationSolution,
                                    baseVelocitySolution,
                                    jointVelocitiesSolution,
                                    worldGravity);
    }

    for (size_t linkIndex = 0; linkIndex < humanModel.getNrOfLinks(); ++linkIndex) {
        std::string linkName = humanModel.getLinkName(linkIndex);

        // skip fake links
        if (wearableStorage.modelToWearable_LinkName.find(linkName)
            == wearableStorage.modelToWearable_LinkName.end()) {
            continue;
        }

        iDynTree::Rotation rotationError =
            computations->getWorldTransform(humanModel.getFrameIndex(linkName)).getRotation()
            * linkTransformMatrices[linkName].getRotation().inverse();
        iDynTree::Vector3 angularVelocityError;

//         angularVelocityError = iDynTreeHelper::Rotation::skewVee(rotationError);
        angularVelocityError = rotationError.log();
        iDynTree::toEigen(integralOrientationError) =
            iDynTree::toEigen(integralOrientationError)
            + iDynTree::toEigen(angularVelocityError) * dt;

        // for floating base link use error also on position if not useDirectBaseMeasurement or useFixedBase,
        // otherwise skip or fix the link.
        if (linkName == floatingBaseFrame) {
            if (useDirectBaseMeasurement) {
                continue;
            }

            if (useFixedBase) {
                linkVelocities[linkName].zero();
                continue;
            }

            iDynTree::Vector3 linearVelocityError;
            linearVelocityError =
                computations->getWorldTransform(humanModel.getFrameIndex(linkName)).getPosition()
                - linkTransformMatrices[linkName].getPosition();

            iDynTree::toEigen(integralLinearVelocityError) =
                iDynTree::toEigen(integralLinearVelocityError)
                + iDynTree::toEigen(linearVelocityError) * dt;
            for (int i = 0; i < 3; i++) {
                linkVelocities[linkName].setVal(i,
                                                integrationBasedIKMeasuredLinearVelocityGain
                                                        * linkVelocities[linkName].getVal(i)
                                                    - integrationBasedIKLinearCorrectionGain
                                                          * linearVelocityError.getVal(i)
                                                    - integrationBasedIKIntegralLinearCorrectionGain
                                                          * integralLinearVelocityError.getVal(i));
            }
        }

        // correct the links angular velocities
        for (int i = 3; i < 6; i++) {
            linkVelocities[linkName].setVal(
                i,
                integrationBasedIKMeasuredAngularVelocityGain * linkVelocities[linkName].getVal(i)
                    - integrationBasedIKAngularCorrectionGain * angularVelocityError.getVal(i - 3)
                    - integrationBasedIKIntegralAngularCorrectionGain
                          * integralOrientationError.getVal(i - 3));
        }

//         if (linkName == floatingBaseFrame && m_extEstimatorInitialized)
//         {
//             for (int idx = 0; idx < 3; idx++) {
//                 linkVelocities[linkName](idx) = baseVelocitySolution(idx);
//             }
//
//         }
    }

    // INVERSE VELOCITY KINEMATICS
    // Set joint configuration
    if (!inverseVelocityKinematics.setConfiguration(baseTransformSolution,
                                                    jointConfigurationSolution)) {
        yError() << LogPrefix
                 << "Failed to set the joint configuration for initializing the global IK";
        return false;
    }

    // Update ivk velocity targets based on wearable input data
    if (!updateInverseVelocityKinematicTargets()) {
        return false;
    }

    if (!inverseVelocityKinematics.solve()) {
        yError() << LogPrefix << "Failed to solve inverse velocity kinematics";
        return false;
    }

    inverseVelocityKinematics.getVelocitySolution(baseVelocitySolution, jointVelocitiesSolution);

    // Threshold to limitate joint velocity
    for (unsigned i = 0; i < jointVelocitiesSolution.size(); i++) {
        if (integrationBasedJointVelocityLimit > 0
            && jointVelocitiesSolution.getVal(i) > integrationBasedJointVelocityLimit) {
            yWarning() << LogPrefix << "joint velocity out of limit: " << humanModel.getJointName(i)
                       << " : " << jointVelocitiesSolution.getVal(i);
            jointVelocitiesSolution.setVal(i, integrationBasedJointVelocityLimit);
        }
        else if (integrationBasedJointVelocityLimit > 0
                 && jointVelocitiesSolution.getVal(i)
                        < (-1.0 * integrationBasedJointVelocityLimit)) {
            yWarning() << LogPrefix << "joint velocity out of limit: " << humanModel.getJointName(i)
                       << " : " << jointVelocitiesSolution.getVal(i);
            jointVelocitiesSolution.setVal(i, -1.0 * integrationBasedJointVelocityLimit);
        }
    }

    // VELOCITY INTEGRATION
    // integrate velocities measurements
    if (!useDirectBaseMeasurement) {
        stateIntegrator.integrate(jointVelocitiesSolution,
                                    baseVelocitySolution.getLinearVec3(),
                                    baseVelocitySolution.getAngularVec3(),
                                    dt);

        stateIntegrator.getJointConfiguration(jointConfigurationSolution);
        stateIntegrator.getBasePose(baseTransformSolution);
    }
    else {
        stateIntegrator.integrate(jointVelocitiesSolutionFiltered, dt);
        stateIntegrator.getJointConfiguration(jointConfigurationSolution);
    }
    return true;
}

bool HumanKinematicEstimator::impl::updateInverseKinematicTargets()
{
    iDynTree::Transform linkTransform;

    for (size_t linkIndex = 0; linkIndex < humanModel.getNrOfLinks(); ++linkIndex) {
        std::string linkName = humanModel.getLinkName(linkIndex);

        // Skip links with no associated measures (use only links from the configuration)
        if (wearableStorage.modelToWearable_LinkName.find(linkName)
            == wearableStorage.modelToWearable_LinkName.end()) {
            continue;
        }

        // For the link used as base insert both the rotation and position cost if not using direcly
        // base measurements and the base is not fixed.
        if (linkName == floatingBaseFrame) {
            if (!(useDirectBaseMeasurement || useFixedBase)) {
                if (!globalIK.updateTarget(linkName, linkTransformMatrices.at(linkName), 1.0, 1.0)) {
                    yError() << LogPrefix << "Failed to update target for floating base" << linkName;
                    return false;
                }
            }
            continue;
        }

        if (linkTransformMatrices.find(linkName) == linkTransformMatrices.end()) {
            yError() << LogPrefix << "Failed to find transformation matrix for link" << linkName;
            return false;
        }

        linkTransform = linkTransformMatrices.at(linkName);
        // if useDirectBaseMeasurement, use the link transform relative to the base
        if (useDirectBaseMeasurement) {
            linkTransform =
                linkTransformMatrices.at(floatingBaseFrame).inverse() * linkTransform;
        }

        if (!globalIK.updateTarget(linkName, linkTransform, posTargetWeight, rotTargetWeight)) {
            yError() << LogPrefix << "Failed to update target for link" << linkName;
            return false;
        }
    }
    return true;
}

bool HumanKinematicEstimator::impl::addInverseKinematicTargets()
{
    for (size_t linkIndex = 0; linkIndex < humanModel.getNrOfLinks(); ++linkIndex) {
        std::string linkName = humanModel.getLinkName(linkIndex);

        // Skip the fake links
        if (wearableStorage.modelToWearable_LinkName.find(linkName)
            == wearableStorage.modelToWearable_LinkName.end()) {
            continue;
        }

        // Insert in the cost the rotation and position of the link used as base, or add it as a target
        if (linkName == floatingBaseFrame) {
            if ((useDirectBaseMeasurement || useFixedBase )
                     && !globalIK.addFrameConstraint(linkName, iDynTree::Transform::Identity())) {
                yError() << LogPrefix << "Failed to add constraint for base link" << linkName;
                return false;
            }
            else if (!globalIK.addTarget(linkName, iDynTree::Transform::Identity(), 1.0, 1.0)) {
                yError() << LogPrefix << "Failed to add target for floating base link" << linkName;
                return false;
            }
            continue;
        }

        // Add ik targets and set to identity
        if (!globalIK.addTarget(
                linkName, iDynTree::Transform::Identity(), posTargetWeight, rotTargetWeight)) {
            yError() << LogPrefix << "Failed to add target for link" << linkName;
            return false;
        }
    }
    return true;
}

bool HumanKinematicEstimator::impl::updateInverseVelocityKinematicTargets()
{
    iDynTree::Twist linkTwist;

    for (size_t linkIndex = 0; linkIndex < humanModel.getNrOfLinks(); ++linkIndex) {
        std::string linkName = humanModel.getLinkName(linkIndex);

        // Skip links with no associated measures (use only links from the configuration)
        if (wearableStorage.modelToWearable_LinkName.find(linkName)
            == wearableStorage.modelToWearable_LinkName.end()) {
            continue;
        }

        // For the link used as base insert both the rotation and position cost if not using direcly
        // measurement from xsens
        if (linkName == floatingBaseFrame) {
            if (!useDirectBaseMeasurement
                && !inverseVelocityKinematics.updateTarget(
                    linkName, linkVelocities.at(linkName), 1.0, 1.0)) {
                yError() << LogPrefix << "Failed to update velocity target for floating base"
                         << linkName;
                return false;
            }
            continue;
        }

        if (linkVelocities.find(linkName) == linkVelocities.end()) {
            yError() << LogPrefix << "Failed to find twist for link" << linkName;
            return false;
        }

        linkTwist = linkVelocities.at(linkName);
        if (useDirectBaseMeasurement) {
            linkTwist = linkTwist - linkVelocities.at(floatingBaseFrame);
        }

        if (!inverseVelocityKinematics.updateTarget(
                linkName, linkTwist, linVelTargetWeight, angVelTargetWeight)) {
            yError() << LogPrefix << "Failed to update velocity target for link" << linkName;
            return false;
        }
    }

    return true;
}

bool HumanKinematicEstimator::impl::addInverseVelocityKinematicsTargets()
{
    for (size_t linkIndex = 0; linkIndex < humanModel.getNrOfLinks(); ++linkIndex) {
        std::string linkName = humanModel.getLinkName(linkIndex);

        // skip the fake links
        if (wearableStorage.modelToWearable_LinkName.find(linkName)
            == wearableStorage.modelToWearable_LinkName.end()) {
            continue;
        }

        // Insert in the cost the twist of the link used as base
        if (linkName == floatingBaseFrame) {
            if (!useDirectBaseMeasurement
                && !inverseVelocityKinematics.addTarget(
                    linkName, iDynTree::Twist::Zero(), 1.0, 1.0)) {
                yError() << LogPrefix << "Failed to add velocity target for floating base link"
                         << linkName;
                return false;
            }
            continue;
        }

        // Add ivk targets and set to zero
        if (!inverseVelocityKinematics.addAngularVelocityTarget(
                linkName, iDynTree::Twist::Zero(), angVelTargetWeight)) {
            yError() << LogPrefix << "Failed to add velocity target for link" << linkName;
            return false;
        }
    }

    return true;
}

bool HumanKinematicEstimator::impl::computeLinksOrientationErrors(
    std::unordered_map<std::string, iDynTree::Transform> linkDesiredTransforms,
    iDynTree::VectorDynSize jointConfigurations,
    iDynTree::Transform floatingBasePose,
    std::unordered_map<std::string, iDynTreeHelper::Rotation::rotationDistance>&
        linkErrorOrientations)
{
    iDynTree::VectorDynSize zeroJointVelocities = jointConfigurations;
    zeroJointVelocities.zero();

    iDynTree::Twist zeroBaseVelocity;
    zeroBaseVelocity.zero();

    iDynTree::KinDynComputations* computations = kinDynComputations.get();
    computations->setRobotState(
        floatingBasePose, jointConfigurations, zeroBaseVelocity, zeroJointVelocities, worldGravity);

    for (const auto& linkMapEntry : linkDesiredTransforms) {
        const ModelLinkName& linkName = linkMapEntry.first;
        linkErrorOrientations[linkName] = iDynTreeHelper::Rotation::rotationDistance(
            computations->getWorldTransform(linkName).getRotation(),
            linkDesiredTransforms[linkName].getRotation());
    }
    return true;
}

bool HumanKinematicEstimator::impl::computeLinksAngularVelocityErrors(
    std::unordered_map<std::string, iDynTree::Twist> linkDesiredVelocities,
    iDynTree::VectorDynSize jointConfigurations,
    iDynTree::Transform floatingBasePose,
    iDynTree::VectorDynSize jointVelocities,
    iDynTree::Twist baseVelocity,
    std::unordered_map<std::string, iDynTree::Vector3>& linkAngularVelocityError)
{
    iDynTree::KinDynComputations* computations = kinDynComputations.get();
    computations->setRobotState(
        floatingBasePose, jointConfigurations, baseVelocity, jointVelocities, worldGravity);

    for (const auto& linkMapEntry : linkDesiredVelocities) {
        const ModelLinkName& linkName = linkMapEntry.first;
        iDynTree::toEigen(linkAngularVelocityError[linkName]) =
            iDynTree::toEigen(linkDesiredVelocities[linkName].getLinearVec3())
            - iDynTree::toEigen(computations->getFrameVel(linkName).getLinearVec3());
    }

    return true;
}

bool HumanKinematicEstimator::attach(yarp::dev::PolyDriver* poly)
{
    if (!poly) {
        yError() << LogPrefix << "Passed PolyDriver is nullptr";
        return false;
    }

    // Get the device name from the driver
    const std::string deviceName = poly->getValue("device").asString();
    std::cerr << "attaching " << deviceName << std::endl;

    if (deviceName == "iwear_remapper") {

        if (pImpl->iWear || !poly->view(pImpl->iWear) || !pImpl->iWear) {
            yError() << LogPrefix << "Failed to view the IWear interface from the PolyDriver";
            return false;
        }

        while (pImpl->iWear->getStatus() == WearStatus::WaitingForFirstRead) {
            yInfo() << LogPrefix << "IWear interface waiting for first data. Waiting...";
            yarp::os::Time::delay(5);
        }

        if (pImpl->iWear->getStatus() != WearStatus::Ok) {
            yError() << LogPrefix << "The status of the attached IWear interface is not ok ("
                    << static_cast<int>(pImpl->iWear->getStatus()) << ")";
            return false;
        }

        // ===========
        // CHECK LINKS
        // ===========

        // Check that the attached IWear interface contains all the model links
        for (size_t linkIndex = 0; linkIndex < pImpl->humanModel.getNrOfLinks(); ++linkIndex) {
            // Get the name of the link from the model and its prefix from iWear
            std::string modelLinkName = pImpl->humanModel.getLinkName(linkIndex);

            if (pImpl->wearableStorage.modelToWearable_LinkName.find(modelLinkName)
                == pImpl->wearableStorage.modelToWearable_LinkName.end()) {
                // yWarning() << LogPrefix << "Failed to find" << modelLinkName
                //           << "entry in the configuration map. Skipping this link.";
                continue;
            }

            // Get the name of the sensor associated to the link
            std::string wearableLinkName =
                pImpl->wearableStorage.modelToWearable_LinkName.at(modelLinkName);

            // Try to get the sensor
            auto sensor = pImpl->iWear->getVirtualLinkKinSensor(wearableLinkName);
            if (!sensor) {
                // yError() << LogPrefix << "Failed to find sensor associated to link" <<
                // wearableLinkName
                //<< "from the IWear interface";
                return false;
            }

            // Create a sensor map entry using the wearable sensor name as key
            pImpl->wearableStorage.linkSensorsMap[wearableLinkName] =
                pImpl->iWear->getVirtualLinkKinSensor(wearableLinkName);
        }

        // ============
        // CHECK JOINTS
        // ============

        if (pImpl->useXsensJointsAngles) {
            yDebug() << "Checking joints";

            for (size_t jointIndex = 0; jointIndex < pImpl->humanModel.getNrOfDOFs(); ++jointIndex) {
                // Get the name of the joint from the model and its prefix from iWear
                std::string modelJointName = pImpl->humanModel.getJointName(jointIndex);

                // Urdfs don't have support of spherical joints, IWear instead does.
                // We use the configuration for addressing this mismatch.
                if (pImpl->wearableStorage.modelToWearable_JointInfo.find(modelJointName)
                    == pImpl->wearableStorage.modelToWearable_JointInfo.end()) {
                    yWarning() << LogPrefix << "Failed to find" << modelJointName
                            << "entry in the configuration map. Skipping this joint.";
                    continue;
                }

                // Get the name of the sensor associate to the joint
                std::string wearableJointName =
                    pImpl->wearableStorage.modelToWearable_JointInfo.at(modelJointName).name;

                // Try to get the sensor
                auto sensor = pImpl->iWear->getVirtualSphericalJointKinSensor(wearableJointName);
                if (!sensor) {
                    yError() << LogPrefix << "Failed to find sensor associated with joint"
                            << wearableJointName << "from the IWear interface";
                    return false;
                }

                // Create a sensor map entry using the wearable sensor name as key
                pImpl->wearableStorage.jointSensorsMap[wearableJointName] = sensor;
            }
        }
    }

    if (deviceName == "human_wrench_provider") {
        // Attach IHumanWrench interfaces coming from HumanWrenchProvider
        if (pImpl->iHumanWrench || !poly->view(pImpl->iHumanWrench) || !pImpl->iHumanWrench) {
            yError() << LogPrefix << "Failed to view iHumanWrench interface from the polydriver";
            return false;
        }

        // Check the interface
        if (pImpl->iHumanWrench->getNumberOfWrenchSources() == 0
                || pImpl->iHumanWrench->getNumberOfWrenchSources() != pImpl->iHumanWrench->getWrenchSourceNames().size()) {
            yError() << "The IHumanWrench interface might not be ready";
            return false;
        }

        // wait for first data
        while (pImpl->iHumanWrench->getWrenches().size() != (pImpl->iHumanWrench->getNumberOfWrenchSources() * 6) ) {
            yInfo() << LogPrefix << "IHumanWrench interface waiting for first data. Waiting...";
            yarp::os::Time::delay(5);
        }

        yInfo() << LogPrefix << deviceName << "human_wrench_provider attach() successful";
    }

    return true;
}

void HumanKinematicEstimator::threadRelease()
{
    if (!pImpl->logData()) {
        yError() << LogPrefix << "Failed to log data";
    }

    if (!pImpl->ikPool->closeIKWorkerPool()) {
        yError() << LogPrefix << "Failed to close the IKWorker pool";
    }
}

bool HumanKinematicEstimator::detach()
{
    while (isRunning()) {
        stop();
    }

    {
        std::lock_guard<std::mutex>(pImpl->mutex);
        pImpl->solution.clear();
    }

    pImpl->iWear = nullptr;
    pImpl->iHumanWrench = nullptr;
    return true;
}

bool HumanKinematicEstimator::detachAll()
{
    return detach();
}

bool HumanKinematicEstimator::attachAll(const yarp::dev::PolyDriverList& driverList)
{
    if (driverList.size() > 2) {
        yError() << LogPrefix << "This wrapper accepts maximum two attached PolyDriver";
        return false;
    }

    bool attachStatus = true;

    for (size_t i = 0; i < driverList.size(); i++) {
        const yarp::dev::PolyDriverDescriptor* driver = driverList[i];

        if (!driver) {
                yError() << LogPrefix << "Passed PolyDriverDescriptor is nullptr";
                return false;
        }
yInfo() << "----> attaching" << driver->key;
            attachStatus = attachStatus && attach(driver->poly);
    }

    // Start the PeriodicThread loop
    if (attachStatus)
    {
        if (!start()) {
            yError() << LogPrefix << "Failed to start the loop.";
            return false;
        }
    }

    return attachStatus;
}

std::vector<std::string> HumanKinematicEstimator::getJointNames() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    std::vector<std::string> jointNames;

    for (size_t jointIndex = 0; jointIndex < pImpl->humanModel.getNrOfJoints(); ++jointIndex) {
        if (pImpl->humanModel.getJoint(jointIndex)->getNrOfDOFs() == 1) {
            jointNames.emplace_back(pImpl->humanModel.getJointName(jointIndex));
        }
    }

    return jointNames;
}

size_t HumanKinematicEstimator::getNumberOfJoints() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->humanModel.getNrOfDOFs();
}

std::string HumanKinematicEstimator::getBaseName() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->floatingBaseFrame;
}

std::vector<double> HumanKinematicEstimator::getJointPositions() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->solution.jointPositions;
}

std::vector<double> HumanKinematicEstimator::getJointVelocities() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->solution.jointVelocities;
}

std::array<double, 6> HumanKinematicEstimator::getBaseVelocity() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->solution.baseVelocity;
}

std::array<double, 4> HumanKinematicEstimator::getBaseOrientation() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->solution.baseOrientation;
}

std::array<double, 3> HumanKinematicEstimator::getBasePosition() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->solution.basePosition;
}
std::array<double, 3> HumanKinematicEstimator::getCoMPosition() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->solution.CoMPosition;
}

std::array<double, 3> HumanKinematicEstimator::getCoMVelocity() const
{
    std::lock_guard<std::mutex> lock(pImpl->mutex);
    return pImpl->solution.CoMVelocity;
}

// This method returns the all link pair names from the full human model
static void createEndEffectorsPairs(
    const iDynTree::Model& model,
    std::vector<SegmentInfo>& humanSegments,
    std::vector<std::pair<std::string, std::string>>& framePairs,
    std::vector<std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex>>& framePairIndeces)
{
    // for each element in human segments
    // extract it from the vector (to avoid duplications)
    // Look for it in the model and get neighbours
    std::vector<SegmentInfo> segments(humanSegments);
    size_t segmentCount = segments.size();

    while (!segments.empty()) {
        SegmentInfo segment = segments.back();
        segments.pop_back();
        segmentCount--;

        iDynTree::LinkIndex linkIndex = model.getLinkIndex(segment.segmentName);
        if (linkIndex < 0 || static_cast<unsigned>(linkIndex) >= model.getNrOfLinks()) {
            yWarning("Segment %s not found in the URDF model", segment.segmentName.c_str());
            continue;
        }

        // this for loop should not be necessary, but this can help keeps the backtrace short
        // as we do not assume that we can go back further that this node
        for (unsigned neighbourIndex = 0; neighbourIndex < model.getNrOfNeighbors(linkIndex);
             ++neighbourIndex) {
            // remember the "biforcations"
            std::vector<iDynTree::LinkIndex> backtrace;
            std::vector<iDynTree::LinkIndex>::iterator Iterator_backtrace;

            // and the visited nodes
            std::vector<iDynTree::LinkIndex> visited;

            // I've already visited the starting node
            visited.push_back(linkIndex);
            iDynTree::Neighbor neighbour = model.getNeighbor(linkIndex, neighbourIndex);
            backtrace.push_back(neighbour.neighborLink);

            while (!backtrace.empty()) {
                iDynTree::LinkIndex currentLink = backtrace.back();
                backtrace.pop_back();
                // add the current link to the visited
                visited.push_back(currentLink);

                std::string linkName = model.getLinkName(currentLink);

                // check if this is a human segment
                std::vector<SegmentInfo>::iterator foundSegment =
                    std::find_if(segments.begin(), segments.end(), [&](SegmentInfo& frame) {
                        return frame.segmentName == linkName;
                    });
                if (foundSegment != segments.end()) {
                    std::vector<SegmentInfo>::difference_type foundLinkIndex =
                        std::distance(segments.begin(), foundSegment);
                    // Found! This is a segment
                    framePairs.push_back(
                        std::pair<std::string, std::string>(segment.segmentName, linkName));
                    framePairIndeces.push_back(
                        std::pair<iDynTree::FrameIndex, iDynTree::FrameIndex>(segmentCount,
                                                                              foundLinkIndex));
                    yInfo() << "Segment : " << segment.segmentName
                            << " , associated neighbor : " << linkName
                            << " , found segment: " << foundSegment->segmentName
                            << " , distance: " << foundLinkIndex;
                    break;
                }

                for (unsigned i = 0; i < model.getNrOfNeighbors(currentLink); ++i) {
                    iDynTree::LinkIndex link = model.getNeighbor(currentLink, i).neighborLink;
                    // check if we already visited this segment
                    if (std::find(visited.begin(), visited.end(), link) != visited.end()) {
                        // Yes => skip
                        continue;
                    }
                    Iterator_backtrace = backtrace.begin();
                    backtrace.insert(Iterator_backtrace, link);
                }
            }
        }
    }
}

static bool getReducedModel(const iDynTree::Model& modelInput,
                            const std::string& parentFrame,
                            const std::string& endEffectorFrame,
                            iDynTree::Model& modelOutput)
{
    iDynTree::FrameIndex parentFrameIndex;
    iDynTree::FrameIndex endEffectorFrameIndex;
    std::vector<std::string> consideredJoints;
    iDynTree::Traversal traversal;
    iDynTree::LinkIndex parentLinkIdx;
    iDynTree::IJointConstPtr joint;
    iDynTree::ModelLoader loader;

    // Get frame indices
    parentFrameIndex = modelInput.getFrameIndex(parentFrame);
    endEffectorFrameIndex = modelInput.getFrameIndex(endEffectorFrame);

    if (parentFrameIndex == iDynTree::FRAME_INVALID_INDEX) {
        yError() << LogPrefix << " Invalid parent frame: " << parentFrame;
        return false;
    }
    else if (endEffectorFrameIndex == iDynTree::FRAME_INVALID_INDEX) {
        yError() << LogPrefix << " Invalid End Effector Frame: " << endEffectorFrame;
        return false;
    }

    // Select joint from traversal
    modelInput.computeFullTreeTraversal(traversal, modelInput.getFrameLink(parentFrameIndex));

    iDynTree::LinkIndex visitedLink = modelInput.getFrameLink(endEffectorFrameIndex);

    while (visitedLink != traversal.getBaseLink()->getIndex()) {
        parentLinkIdx = traversal.getParentLinkFromLinkIndex(visitedLink)->getIndex();
        joint = traversal.getParentJointFromLinkIndex(visitedLink);

        // Check if the joint is supported
        if (modelInput.getJoint(joint->getIndex())->getNrOfDOFs() == 1) {
            consideredJoints.insert(consideredJoints.begin(),
                                    modelInput.getJointName(joint->getIndex()));
        }
        else {
            yWarning() << LogPrefix << "Joint " << modelInput.getJointName(joint->getIndex())
                       << " is ignored as it has ("
                       << modelInput.getJoint(joint->getIndex())->getNrOfDOFs() << " DOFs)";
        }

        visitedLink = parentLinkIdx;
    }

    if (!loader.loadReducedModelFromFullModel(modelInput, consideredJoints)) {
        yWarning() << LogPrefix << " failed to select joints: ";
        for (std::vector<std::string>::const_iterator i = consideredJoints.begin();
             i != consideredJoints.end();
             ++i) {
            yWarning() << *i << ' ';
        }
        return false;
    }

    modelOutput = loader.model();

    return true;
}
