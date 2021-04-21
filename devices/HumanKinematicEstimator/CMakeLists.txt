# Copyright (C) 2018 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

find_package(IWear REQUIRED)
find_package(iDynTree REQUIRED)
find_package(OsqpEigen 0.4.0 REQUIRED)
find_package(ICUB REQUIRED)
find_package(manif REQUIRED)
find_package(matioCpp REQUIRED)
find_package(BipedalLocomotionFramework REQUIRED)

yarp_prepare_plugin(human_kinematic_estimator
    TYPE hde::devices::HumanKinematicEstimator
    INCLUDE HumanKinematicEstimator.h
    CATEGORY device
    ADVANCED
    DEFAULT ON)

yarp_add_plugin(HumanKinematicEstimator
    HumanKinematicEstimator.cpp
    HumanKinematicEstimator.h
    IKWorkerPool.cpp
    IKWorkerPool.h
    InverseVelocityKinematics/InverseVelocityKinematics.cpp
    InverseVelocityKinematics/InverseVelocityKinematics.hpp)

target_include_directories(HumanKinematicEstimator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})

target_link_libraries(HumanKinematicEstimator PUBLIC
    IHumanState
    IHumanWrench
    YARP::YARP_OS
    YARP::YARP_dev
    YARP::YARP_init
    IWear::IWear
    iDynTree::idyntree-model
    iDynTree::idyntree-inverse-kinematics
    iDynTree::idyntree-high-level
    UtilityLibrary
    OsqpEigen::OsqpEigen
    osqp::osqp
    ctrlLib
    BipedalLocomotion::ParametersHandlerYarpImplementation
    BipedalLocomotion::FloatingBaseEstimators
    BipedalLocomotion::ContactDetectors
    BipedalLocomotion::matioCppConversions
    )

if(MSVC)
  target_compile_options(HumanKinematicEstimator PRIVATE /bigobj)
endif()

yarp_install(
    TARGETS HumanKinematicEstimator
    COMPONENT runtime
    LIBRARY DESTINATION ${YARP_DYNAMIC_PLUGINS_INSTALL_DIR}
    ARCHIVE DESTINATION ${YARP_STATIC_PLUGINS_INSTALL_DIR}
    YARP_INI DESTINATION ${YARP_PLUGIN_MANIFESTS_INSTALL_DIR})
