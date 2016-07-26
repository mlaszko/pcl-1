/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#ifndef PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_ELLIPSE_H_
#define PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_ELLIPSE_H_
#include <pcl/sample_consensus/sac_model_ellipse.h>
#include <pcl/sample_consensus/eigen.h>
#include <pcl/common/concatenate.h>
#include <cmath>

//////////////////////////////////////////////////////////////////////////
template <typename PointT> bool
pcl::SampleConsensusModelEllipse2D<PointT>::isSampleGood(const std::vector<int> &samples) const
{
    Eigen::Vector2d p0 (input_->points[samples[0]].x, input_->points[samples[0]].y);
    Eigen::Vector2d p1 (input_->points[samples[1]].x, input_->points[samples[1]].y);
    Eigen::Vector2d p2 (input_->points[samples[2]].x, input_->points[samples[2]].y);
    Eigen::Vector2d p3 (input_->points[samples[3]].x, input_->points[samples[3]].y);
    Eigen::Vector2d p4 (input_->points[samples[4]].x, input_->points[samples[4]].y);

//    cout << p0 << endl << p1 << endl << p2 << endl << p3 << endl << p4 << endl;

    Eigen::MatrixXd  A(5,5);

    A.row(0) << p0[1]*p0[1] , p0[0]*p0[1] , p0[0] , p0[1] , 1 ;
    A.row(1) << p1[1]*p1[1] , p1[0]*p1[1] , p1[0] , p1[1] , 1 ;
    A.row(2) << p2[1]*p2[1] , p2[0]*p2[1] , p2[0] , p2[1] , 1 ;
    A.row(3) << p3[1]*p3[1] , p3[0]*p3[1] , p3[0] , p3[1] , 1 ;
    A.row(4) << p4[1]*p4[1] , p4[0]*p4[1] , p4[0] , p4[1] , 1 ;

//    cout << A << endl;


    Eigen::Matrix<double, 5, 1> b;
    b << -p0[0]*p0[0] , -p1[0]*p1[0] , -p2[0]*p2[0] , -p3[0]*p3[0] , -p4[0]*p4[0] ;

//    cout << b << endl;

    Eigen::Matrix<double, 5, 1> x = A.colPivHouseholderQr().solve(b);
//    Eigen::Matrix<double, 6, 1> x = A.fullPivLu().solve(z);
//    Eigen::Matrix<double, 6, 1> x = A.fullPivHouseholderQr().solve(z);
//    Eigen::Matrix<double, 6, 1> x = A.jacobiSvd().solve(z);
//    Eigen::Matrix<double, 6, 1> x = A.householderQr().solve(z);
//    cout<< "1 " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] << " " << x[4] << " " << endl;

    bool poprawny = 4*x[0] - x[1]*x[1] > 0;
//    if(!poprawny)
//        cout<< "Niepoprawna elipsa " << 4*x[0] - x[1]*x[1] << endl;
//    else
//        cout<< 4*x[1] - x[0]*x[0] << endl;

//    cout<<" Poprawne " << poprawne <<endl;


    return poprawny;
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT> bool
pcl::SampleConsensusModelEllipse2D<PointT>::computeModelCoefficients (const std::vector<int> &samples, Eigen::VectorXf &model_coefficients)
{
  // Need 5 samples
  if (samples.size () != 5)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelEllipse2D::computeModelCoefficients] Invalid set of samples given (%lu)!\n", samples.size ());
    return (false);
  }

  model_coefficients.resize (5);

  Eigen::Vector2d p0 (input_->points[samples[0]].x, input_->points[samples[0]].y);
  Eigen::Vector2d p1 (input_->points[samples[1]].x, input_->points[samples[1]].y);
  Eigen::Vector2d p2 (input_->points[samples[2]].x, input_->points[samples[2]].y);
  Eigen::Vector2d p3 (input_->points[samples[3]].x, input_->points[samples[3]].y);
  Eigen::Vector2d p4 (input_->points[samples[4]].x, input_->points[samples[4]].y);

  Eigen::MatrixXd  A(5,5);

  A.row(0) << p0[1]*p0[1] , p0[0]*p0[1] , p0[0] , p0[1] , 1 ;
  A.row(1) << p1[1]*p1[1] , p1[0]*p1[1] , p1[0] , p1[1] , 1 ;
  A.row(2) << p2[1]*p2[1] , p2[0]*p2[1] , p2[0] , p2[1] , 1 ;
  A.row(3) << p3[1]*p3[1] , p3[0]*p3[1] , p3[0] , p3[1] , 1 ;
  A.row(4) << p4[1]*p4[1] , p4[0]*p4[1] , p4[0] , p4[1] , 1 ;


  Eigen::Matrix<double, 5, 1> b;
  b << -p0[0]*p0[0] , -p1[0]*p1[0] , -p2[0]*p2[0] , -p3[0]*p3[0] , -p4[0]*p4[0] ;

  Eigen::Matrix<double, 5, 1> x = A.colPivHouseholderQr().solve(b);

//  if(4*x[0] - x[1]*x[1] <= 0)
//      cout<<"Błąd!!!\n";
//  else
//      poprawne++;

  double tan2fi = x[1]/(1-x[0]);
  double fi = atan(tan2fi)/2;

  double sinfi = sin(fi);
  double cosfi = cos(fi);
  double sin2fi = sinfi * sinfi;
  double cos2fi = cosfi * cosfi;

  //A=1
  double ap = cos2fi + x[1]*cosfi*sinfi + x[0]*sin2fi;
  double cp = sin2fi - x[1]*cosfi*sinfi + x[0]*cos2fi;
  double dp = x[2]*cosfi  + x[3]*sinfi;
  double ep = -x[2]*sinfi + x[3]*cosfi;
  double fp = x[4];

  double x0p = -dp/2*ap;
  double y0p = -ep/2*cp;
  double a2 = (-4*fp*ap*cp +cp*dp*dp + ap*ep*ep) / (4*ap*cp*cp);
  double b2 = (-4*fp*ap*cp +cp*dp*dp + ap*ep*ep) / (4*ap*ap*cp);

  double x0 = x0p*cosfi - y0p*sinfi;
  double y0 = x0p*sinfi + y0p*cosfi;

  // Center (x, y)
  model_coefficients[0] = x0;
  model_coefficients[1] = y0;

  // Radius
  model_coefficients[2] = sqrt(a2);
  model_coefficients[3] = sqrt(b2);

  model_coefficients[4] = fi;

//  cout << x0 << " " << y0 << " " << sqrt(a2) << " " << sqrt(b2) << " " << fi << endl;
  return (true);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SampleConsensusModelEllipse2D<PointT>::getDistancesToModel (const Eigen::VectorXf &model_coefficients, std::vector<double> &distances)
{
  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
    distances.clear ();
    return;
  }
  distances.resize (indices_->size ());

  Eigen::Vector2d C (model_coefficients[0], model_coefficients[1]);
  double a = model_coefficients[2];
  double b = model_coefficients[3];
  double fi = model_coefficients[4];

  // Iterate through the 3d points and calculate the distances from them to the ellipse
  for (size_t i = 0; i < indices_->size (); ++i){
      Eigen::Vector2d P(input_->points[(*indices_)[i]].x,
                        input_->points[(*indices_)[i]].y) ;


      Eigen::Vector2d helperVectorPC = P - C;
      double sinth = helperVectorPC[1] / helperVectorPC.norm();
      double costh = helperVectorPC[0] / helperVectorPC.norm();
      double Xk = C[0] + a*costh*cos(fi) - b*sinth*sin(fi);
      double Yk = C[1] + a*costh*sin(fi) + b*sinth*cos(fi);
      Eigen::Vector2d K ( Xk, Yk);
      Eigen::Vector2d distanceVector =  P - K;

      distances[i] = distanceVector.norm();
  }

}

//////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SampleConsensusModelEllipse2D<PointT>::selectWithinDistance (
    const Eigen::VectorXf &model_coefficients, const double threshold,
    std::vector<int> &inliers)
{
  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
  {
    inliers.clear ();
    return;
  }
  int nr_p = 0;
  inliers.resize (indices_->size ());
  error_sqr_dists_.resize (indices_->size ());

  Eigen::Vector2d C (model_coefficients[0], model_coefficients[1]);
  double a = model_coefficients[2];
  double b = model_coefficients[3];
  double fi = model_coefficients[4];

  // Iterate through the 3d points and calculate the distances from them to the sphere
  for (size_t i = 0; i < indices_->size (); ++i)
  {
    // Calculate the distance from the point to the ellipse
    Eigen::Vector2d P(input_->points[(*indices_)[i]].x,
                      input_->points[(*indices_)[i]].y) ;


    Eigen::Vector2d helperVectorPC = P - C;
    double sinth = helperVectorPC[1] / helperVectorPC.norm();
    double costh = helperVectorPC[0] / helperVectorPC.norm();
    double Xk = C[0] + a*costh*cos(fi) - b*sinth*sin(fi);
    double Yk = C[1] + a*costh*sin(fi) + b*sinth*cos(fi);
    Eigen::Vector2d K ( Xk, Yk);
    Eigen::Vector2d distanceVector =  P - K;

    float distance = distanceVector.norm();

    if (distance < threshold)
    {
      // Returns the indices of the points whose distances are smaller than the threshold
      inliers[nr_p] = (*indices_)[i];
      error_sqr_dists_[nr_p] = static_cast<double> (distance);
      ++nr_p;
    }
  }
  inliers.resize (nr_p);
  error_sqr_dists_.resize (nr_p);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT> int
pcl::SampleConsensusModelEllipse2D<PointT>::countWithinDistance (
    const Eigen::VectorXf &model_coefficients, const double threshold)
{
  // Check if the model is valid given the user constraints
  if (!isModelValid (model_coefficients))
    return (0);
  int nr_p = 0;
  Eigen::Vector2d C (model_coefficients[0], model_coefficients[1]);
  double a = model_coefficients[2];
  double b = model_coefficients[3];
  double fi = model_coefficients[4];

  // Iterate through the 3d points and calculate the distances from them to the sphere
  for (size_t i = 0; i < indices_->size (); ++i)
  {
    // Calculate the distance from the point to the sphere as the difference between //TODO opis
    // dist(point,sphere_origin) and sphere_radius
    Eigen::Vector2d P(input_->points[(*indices_)[i]].x,
                    input_->points[(*indices_)[i]].y) ;
    Eigen::Vector2d helperVectorPC = P - C;
    double sinth = helperVectorPC[1] / helperVectorPC.norm();
    double costh = helperVectorPC[0] / helperVectorPC.norm();
    double Xk = C[0] + a*costh*cos(fi) - b*sinth*sin(fi);
    double Yk = C[1] + a*costh*sin(fi) + b*sinth*cos(fi);
    Eigen::Vector2d K ( Xk, Yk);
    Eigen::Vector2d distanceVector =  P - K;

    if (distanceVector.norm() < threshold)
      nr_p++;
  }
  return (nr_p);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::SampleConsensusModelEllipse2D<PointT>::optimizeModelCoefficients (
      const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients, Eigen::VectorXf &optimized_coefficients)
{
  optimized_coefficients = model_coefficients;

  // Needs a set of valid model coefficients
  if (model_coefficients.size () != 5)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelEllipse2D::optimizeModelCoefficients] Invalid number of model coefficients given (%lu)!\n", model_coefficients.size ());
    return;
  }

  // Need at least 5 samples
  if (inliers.size () <= 5)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelEllipse2D::optimizeModelCoefficients] Not enough inliers found to support a model (%lu)! Returning the same coefficients.\n", inliers.size ());
    return;
  }

  tmp_inliers_ = &inliers;

  OptimizationFunctor functor (static_cast<int> (inliers.size ()), this);
  Eigen::NumericalDiff<OptimizationFunctor> num_diff (functor);
  Eigen::LevenbergMarquardt<Eigen::NumericalDiff<OptimizationFunctor>, float> lm (num_diff);
  int info = lm.minimize (optimized_coefficients);

  // Compute the L2 norm of the residuals
  PCL_DEBUG ("[pcl::SampleConsensusModelEllipse2D::optimizeModelCoefficients] LM solver finished with exit code %i, having a residual norm of %g. \nInitial solution: %g %g %g %g %g\nFinal solution: %g %g %g %g %g\n",
             info, lm.fvec.norm (),
             model_coefficients[0], model_coefficients[1], model_coefficients[2], model_coefficients[3], model_coefficients[4],
             optimized_coefficients[0], optimized_coefficients[1], optimized_coefficients[2], optimized_coefficients[3], optimized_coefficients[4]);
}

//////////////////////////////////////////////////////////////////////////TODO
template <typename PointT> void
pcl::SampleConsensusModelEllipse2D<PointT>::projectPoints (
      const std::vector<int> &inliers, const Eigen::VectorXf &model_coefficients,
      PointCloud &projected_points, bool copy_data_fields)
{
  // Needs a valid set of model coefficients
  if (model_coefficients.size () != 5)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelEllipse2D::projectPoints] Invalid number of model coefficients given (%lu)!\n", model_coefficients.size ());
    return;
  }

  projected_points.header   = input_->header;
  projected_points.is_dense = input_->is_dense;

  // Copy all the data fields from the input cloud to the projected one?
  if (copy_data_fields)
  {
    // Allocate enough space and copy the basics
    projected_points.points.resize (input_->points.size ());
    projected_points.width    = input_->width;
    projected_points.height   = input_->height;

    typedef typename pcl::traits::fieldList<PointT>::type FieldList;
    // Iterate over each point
    for (size_t i = 0; i < projected_points.points.size (); ++i)
      // Iterate over each dimension
      pcl::for_each_type <FieldList> (NdConcatenateFunctor <PointT, PointT> (input_->points[i], projected_points.points[i]));


    Eigen::Vector2d C (model_coefficients[0], model_coefficients[1]);
    double a = model_coefficients[2];
    double b = model_coefficients[3];
    double fi = model_coefficients[4];

    // Iterate through the 3d points and calculate the distances from them to the plane
    for (size_t i = 0; i < inliers.size (); ++i)
    {
        Eigen::Vector2d P(input_->points[inliers[i]].x,
                          input_->points[inliers[i]].y) ;
        Eigen::Vector2d helperVectorPC = P - C;
        double sinth = helperVectorPC[1] / helperVectorPC.norm();
        double costh = helperVectorPC[0] / helperVectorPC.norm();
        double Xk = C[0] + a*costh*cos(fi) - b*sinth*sin(fi);
        double Yk = C[1] + a*costh*sin(fi) + b*sinth*cos(fi);
        Eigen::Vector2d K ( Xk, Yk);
        Eigen::Vector2d distanceVector =  P - K;

//        if (distanceVector.norm() < threshold)
//      float dx = input_->points[inliers[i]].x - model_coefficients[0];
//      float dy = input_->points[inliers[i]].y - model_coefficients[1];
//      float a = sqrtf ( (model_coefficients[2] * model_coefficients[2]) / (dx * dx + dy * dy) );

      projected_points.points[inliers[i]].x = Xk;
      projected_points.points[inliers[i]].y = Yk;
    }
  }
  else
  {
    // Allocate enough space and copy the basics
    projected_points.points.resize (inliers.size ());
    projected_points.width    = static_cast<uint32_t> (inliers.size ());
    projected_points.height   = 1;

    typedef typename pcl::traits::fieldList<PointT>::type FieldList;
    // Iterate over each point
    for (size_t i = 0; i < inliers.size (); ++i)
      // Iterate over each dimension
      pcl::for_each_type <FieldList> (NdConcatenateFunctor <PointT, PointT> (input_->points[inliers[i]], projected_points.points[i]));

    Eigen::Vector2d C (model_coefficients[0], model_coefficients[1]);
    double a = model_coefficients[2];
    double b = model_coefficients[3];
    double fi = model_coefficients[4];
    // Iterate through the 3d points and calculate the distances from them to the plane
    for (size_t i = 0; i < inliers.size (); ++i)
    {
        Eigen::Vector2d P(input_->points[inliers[i]].x,
                          input_->points[inliers[i]].y) ;
        Eigen::Vector2d helperVectorPC = P - C;
        double sinth = helperVectorPC[1] / helperVectorPC.norm();
        double costh = helperVectorPC[0] / helperVectorPC.norm();
        double Xk = C[0] + a*costh*cos(fi) - b*sinth*sin(fi);
        double Yk = C[1] + a*costh*sin(fi) + b*sinth*cos(fi);
        Eigen::Vector2d K ( Xk, Yk);
        Eigen::Vector2d distanceVector =  P - K;

        projected_points.points[inliers[i]].x = Xk;
        projected_points.points[inliers[i]].y = Yk;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT> bool
pcl::SampleConsensusModelEllipse2D<PointT>::doSamplesVerifyModel (
      const std::set<int> &indices, const Eigen::VectorXf &model_coefficients, const double threshold)
{
  // Needs a valid model coefficients
  if (model_coefficients.size () != 5)
  {
    PCL_ERROR ("[pcl::SampleConsensusModelEllipse2D::doSamplesVerifyModel] Invalid number of model coefficients given (%lu)!\n", model_coefficients.size ());
    return (false);
  }
  double a = model_coefficients[2];
  double b = model_coefficients[3];
  double fi = model_coefficients[4];
  Eigen::Vector2d C (model_coefficients[0], model_coefficients[1]);
  for (std::set<int>::const_iterator it = indices.begin (); it != indices.end (); ++it)
  {
    // Calculate the distance from the point to the sphere as the difference between
    //dist(point,sphere_origin) and sphere_radius
    Eigen::Vector2d P(input_->points[*it].x,
                      input_->points[*it].y) ;
    Eigen::Vector2d helperVectorPC = P - C;
    double sinth = helperVectorPC[1] / helperVectorPC.norm();
    double costh = helperVectorPC[0] / helperVectorPC.norm();
    double Xk = C[0] + a*costh*cos(fi) - b*sinth*sin(fi);
    double Yk = C[1] + a*costh*sin(fi) + b*sinth*cos(fi);
    Eigen::Vector2d K ( Xk, Yk);
    Eigen::Vector2d distanceVector =  P - K;

    if ( distanceVector.norm() > threshold)
      return (false);
  }

  return (true);
}

//////////////////////////////////////////////////////////////////////////
template <typename PointT> bool 
pcl::SampleConsensusModelEllipse2D<PointT>::isModelValid (const Eigen::VectorXf &model_coefficients)
{
  if (!SampleConsensusModel<PointT>::isModelValid (model_coefficients))
    return (false);

//  if (radius_min_ != -std::numeric_limits<double>::max() && model_coefficients[2] < radius_min_)
//    return (false);
//  if (radius_max_ != std::numeric_limits<double>::max() && model_coefficients[2] > radius_max_)
//    return (false);
//  if (radius_min_ != -std::numeric_limits<double>::max() && model_coefficients[3] < radius_min_)
//    return (false);
//  if (radius_max_ != std::numeric_limits<double>::max() && model_coefficients[3] > radius_max_)
//    return (false);

  return (true);
}

#define PCL_INSTANTIATE_SampleConsensusModelEllipse2D(T) template class PCL_EXPORTS pcl::SampleConsensusModelEllipse2D<T>;

#endif    // PCL_SAMPLE_CONSENSUS_IMPL_SAC_MODEL_ELLIPSE_H_

