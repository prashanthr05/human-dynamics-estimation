/*
 * Copyright (C) 2018 Istituto Italiano di Tecnologia (IIT)
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms of the
 * GNU Lesser General Public License v2.1 or any later version.
 */

#include <WrenchFrameTransformers.h>
#include <iDynTree/Core/EigenHelpers.h>
#include <iDynTree/Model/Link.h>

using namespace hde::devices::impl;

bool FixedFrameWrenchTransformer::transformWrenchFrame(const iDynTree::Wrench inputWrench,
                                                       iDynTree::Wrench& transformedWrench)
{
    Eigen::Matrix<double,6,1> transformedWrenchEigen = iDynTree::toEigen(transform.asAdjointTransformWrench())
                                                       * iDynTree::toEigen(inputWrench.asVector());
    iDynTree::fromEigen(transformedWrench, transformedWrenchEigen);
    return true;
}

bool RobotFrameWrenchTransformer::transformWrenchFrame(const iDynTree::Wrench inputWrench,
                                                       iDynTree::Wrench& transformedWrench)
{
    std::lock_guard<std::mutex> lock(_mutex);

    Eigen::Matrix<double,6,1> transformedWrenchEigen = iDynTree::toEigen(transform.asAdjointTransformWrench())
                                                       * iDynTree::toEigen(inputWrench.asVector());
    iDynTree::fromEigen(transformedWrench, transformedWrenchEigen);
    return true;
}
