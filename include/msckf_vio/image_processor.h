/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 *
 * @note msckf_vio中的T_a_b表示a到b的变换矩阵
 * @date 2020-12-12
 */

#ifndef MSCKF_VIO_IMAGE_PROCESSOR_H
#define MSCKF_VIO_IMAGE_PROCESSOR_H

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#define WITH_LIFETIME_STATISTICS 0
#define WITH_LC 0

namespace msckf_vio {

    /**
     * @brief ImageProcessor Detects and tracks features in image sequences.
     */
    class ImageProcessor {
    public:
        // Constructor
        ImageProcessor(ros::NodeHandle &n);
        // Destructor
        ~ImageProcessor();

        // Disable copy and assign constructors.
        ImageProcessor(const ImageProcessor &) = delete;
        ImageProcessor operator=(const ImageProcessor &) = delete;

        // Initialize the object.
        bool initialize();

        typedef std::shared_ptr<ImageProcessor> Ptr;
        typedef std::shared_ptr<const ImageProcessor> ConstPtr;

    private:

        /**
         * @brief ProcessorConfig Configuration parameters for feature detection and tracking.
         */
        struct ProcessorConfig {
            int grid_row;               // 均衡视觉特征点时划分的格子区间的行数 
            int grid_col;               // 均衡视觉特征点时划分的格子区间的列数
            int grid_min_feature_num;
            int grid_max_feature_num;

            int pyramid_levels;         // 图像金字塔的层数
            int patch_size;             // window size of optical flow algorithm
            int fast_threshold;         // OpenCV中fast角点检测阈值
            int max_iteration;          // KLT算法最高迭代次数
            double track_precision;     // KLT算法跟踪精度
            double ransac_threshold;    // 2-points Ransac中内点阈值（像素）
            double stereo_threshold;
        };

        /**
         * @brief FeatureIDType An alias for unsigned long long int.
         */
        typedef unsigned long long int FeatureIDType;

        /**
         * @brief FeatureMetaData Contains necessary information of a feature for easy access.
         */
        struct FeatureMetaData {
            FeatureIDType id;
            float response;
            int lifetime;
            cv::Point2f cam0_point;
            cv::Point2f cam1_point;
        };

        /**
         * @brief GridFeatures
         *    Organize features based on the grid they belong to. Note that the key is encoded by the grid index.
         */
        typedef std::map<int, std::vector<FeatureMetaData> > GridFeatures;

        /**
         * @brief keyPointCompareByResponse
         *    Compare two keypoints based on the response.
         */
        static bool keyPointCompareByResponse(const cv::KeyPoint &pt1, const cv::KeyPoint &pt2) {
            // Keypoint with higher response will be at the beginning of the vector.
            return pt1.response > pt2.response;
        }

        /**
         * @brief featureCompareByResponse
         *    Compare two features based on the response.
         */
        static bool featureCompareByResponse(const FeatureMetaData &f1, const FeatureMetaData &f2) {
            // Features with higher response will be at the beginning of the vector.
            return f1.response > f2.response;
        }

        /**
         * @brief featureCompareByLifetime
         *    Compare two features based on the lifetime.
         */
        static bool featureCompareByLifetime(const FeatureMetaData &f1, const FeatureMetaData &f2) {
            // Features with longer lifetime will be at the beginning of the vector.
            return f1.lifetime > f2.lifetime;
        }

        /**
         * @brief loadParameters
         *    Load parameters from the parameter server.
         */
        bool loadParameters();

        /**
         * @brief createRosIO
         *    Create ros publisher and subscirbers.
         */
        bool createRosIO();

        /**
         * @brief stereoCallback
         *    Callback function for the stereo images.
         * @details 接收双目图像，进行双目特征跟踪，先光流跟踪上一帧的特征点，然后提取新的特征点。将跟踪到的特征点publish出去，供后端接收使用。
         * @param cam0_img left image.
         * @param cam1_img right image.
         */
        void stereoCallback(
                const sensor_msgs::ImageConstPtr &cam0_img,
                const sensor_msgs::ImageConstPtr &cam1_img);

        /**
         * @brief imuCallback
         *    Callback function for the imu message.
         * @details 接收IMU数据，将IMU数据存到imu_msg_buffer中，这里并不处理数据
         * @param msg IMU msg.
         */
        void imuCallback(const sensor_msgs::ImuConstPtr &msg);

        /**
         * @initializeFirstFrame
         *    Initialize the image processing sequence, which is
         *    bascially detect new features on the first set of
         *    stereo images.
         */
        void initializeFirstFrame();

        /**
         * @brief Tracker features on the newly received stereo images.
         *
         * 1. 左图特征点前后帧跟踪，得到当前帧左图特征点;
         *    当前帧左右图跟踪，得到当前帧右图特征点;
         *    左右图分别做前后帧RANSAC剔除外点;
         *    左右图上最后的特征点数量是相同的!
         *
         * 2. 前后帧跟踪和左右图跟踪都是用的LK光流;
         *    前后帧跟踪会用IMU积分的相对旋转预测特征点在当前帧的位置作为初值(integrateImuData, predictFeatureTracking);
         *    左右图跟踪会用相机外参预测右图特征点位置作为初值;
         *
         * 3. 将图像分成了4*5个网格(grid)，每个网格中最多4个特征点，这样能够使特征点均匀分布在图像上
         *
         * Note: 此函数先使用IMU积分预测cam0前后两帧的旋转，KLT光流跟踪cam0上特征点，remove pre帧cam0、cam1以及cur帧cam0的外点;
         *       使用相机外参预测cur帧cam1上特征点，左右眼KLT，remove pre帧cam0、cam1以及cur帧cam0、cam1的外点;
         *       cam0、cam1分别做前后两帧twoPointRansac，得到同维度的cam0_ransac_inliers，cam1_ransac_inliers;
         * 这里的外点去除也可借助cv::findFundamentalMat等函数完成？
         */
        void trackFeatures();


        /**
         * @brief Detect new features on the image to ensure that the features are uniformly distributed on the image. 
         */
        void addNewFeatures();

        /**
         * @brief pruneGridFeatures
         *    Remove some of the features of a grid in case there are
         *    too many features inside of that grid, which ensures the
         *    number of features within each grid is bounded.
         */
        void pruneGridFeatures();

        /**
         * @brief publish
         *    Publish the features on the current image including
         *    both the tracked and newly detected ones.
         */
        void publish();

        /**
         * @brief drawFeaturesMono
         *    Draw tracked and newly detected features on the left
         *    image only.
         */
        void drawFeaturesMono();

        /**
         * @brief drawFeaturesStereo
         *    Draw tracked and newly detected features on the stereo images.
         */
        void drawFeaturesStereo();

        /**
         * @brief createImagePyramids
         *    Create image pyramids used for klt tracking.
         */
        void createImagePyramids();



        /**
         * @brief  Integrates the IMU gyro readings between the two consecutive images, which is used for both tracking prediction and 2-point RANSAC.
         *
         * 1. 用前后两帧图像之间的IMU数据，积分计算帧间图像的相对旋转cam0_R_p_c，cam1_R_p_c
         * 2. 直接使用两帧图像之间IMU角速度的均值 * dt来近似
         *
         * @param[out] cam0_R_p_c: a rotation matrix which takes a vector from previous cam0 frame to current cam0 frame.
         * @param[out] cam1_R_p_c: a rotation matrix which takes a vector from previous cam1 frame to current cam1 frame.
         */
        void integrateImuData(cv::Matx33f &cam0_R_p_c, cv::Matx33f &cam1_R_p_c);


        /**
         * @brief      Compensates the rotation between consecutive camera frames so that feature tracking would be more robust and fast.
         *
         * @param[in]  input_pts: features in the previous image to be tracked.
         * @param[in]  R_p_c: a rotation matrix takes a vector in the previous camera frame to the current camera frame.
         * @param[in]  intrinsics: intrinsic matrix of the camera.
         * @param[out] compensated_pts: predicted locations of the features in the current image based on the provided rotation.
         *
         * Note that the input and output points are of pixel coordinates.
         */
        void predictFeatureTracking(
                const std::vector<cv::Point2f> &input_pts,
                const cv::Matx33f &R_p_c,
                const cv::Vec4d &intrinsics,
                std::vector<cv::Point2f> &compenstated_pts);

        /**
         * @brief      Applies two point ransac algorithm to mark the inliers in the input set.
         *
         * @param[in]  pts1                 previous set of points.
         * @param[in]  pts2                 current set of points.              
         * @param[in]  R_p_c                a rotation matrix takes a vector in the previous camera frame to the current camera frame.
         * @param[in]  intrinsics           intrinsics of the camera.
         * @param[in]  distortion_model     distortion model of the camera.
         * @param[in]  distortion_coeffs    distortion coeffs.
         * @param[in]  inlier_error         acceptable error to be considered as an inlier.
         * @param[in]  success_probability  the required probability of success. Ransac算法的置信度
         * @param[out] inlier_markers       1 for inliers and 0 for outliers.
         */
        void twoPointRansac(
                const std::vector<cv::Point2f> &pts1,
                const std::vector<cv::Point2f> &pts2,
                const cv::Matx33f &R_p_c,
                const cv::Vec4d &intrinsics,
                const std::string &distortion_model,
                const cv::Vec4d &distortion_coeffs,
                const double &inlier_error,
                const double &success_probability,
                std::vector<int> &inlier_markers);
        /**
         * @brief      去除pts_in的畸变
         *
         * @param[in]  pts_in                The points in
         * @param[in]  intrinsics            The intrinsics
         * @param[in]  distortion_model      The distortion model
         * @param[in]  distortion_coeffs     The distortion coeffs
         * @param[out] pts_out               The points out
         * @param[in]  rectification_matrix  The rectification matrix
         * @param[in]  new_intrinsics        The new intrinsics
         *
         * Note 结合rectification_matrix也可以投影特征点，故此函数用法不单一
         */
        void undistortPoints(
                const std::vector<cv::Point2f> &pts_in,
                const cv::Vec4d &intrinsics,
                const std::string &distortion_model,
                const cv::Vec4d &distortion_coeffs,
                std::vector<cv::Point2f> &pts_out,
                const cv::Matx33d &rectification_matrix = cv::Matx33d::eye(),
                const cv::Vec4d &new_intrinsics = cv::Vec4d(1, 1, 0, 0));

        void rescalePoints(
                std::vector<cv::Point2f> &pts1,
                std::vector<cv::Point2f> &pts2,
                float &scaling_factor);

        /**
         * @brief      将归一化平面上的特征点投影到像素平面（加上了畸变）
         *
         * @param[in]  pts_in             The points in
         * @param[in]  intrinsics         The intrinsics
         * @param[in]  distortion_model   The distortion model
         * @param[in]  distortion_coeffs  The distortion coeffs
         *
         * @return     像素平面的特征点 
         */
        std::vector<cv::Point2f> distortPoints(
                const std::vector<cv::Point2f> &pts_in,
                const cv::Vec4d &intrinsics,
                const std::string &distortion_model,
                const cv::Vec4d &distortion_coeffs);


        /**
         * @brief       Matches features with stereo image pairs.
         *
         * @param[in]   cam0_points: points in the left image.
         * @param[out]  cam1_points: points in the right image.
         * @param[out]  inlier_markers: 1 if the match is valid, 0 otherwise.
         *
         * Note 此函数并未按照inlier_markers去除外点
         */
        void stereoMatch(
                const std::vector<cv::Point2f> &cam0_points,
                std::vector<cv::Point2f> &cam1_points,
                std::vector<unsigned char> &inlier_markers);

        /**
         * @brief removeUnmarkedElements Remove the unmarked elements within a vector.
         * @param raw_vec: vector with outliers.
         * @param markers: 0 will represent a outlier, 1 will be an inlier.
         * @return refined_vec: a vector without outliers.
         *
         * Note that the order of the inliers in the raw_vec is perserved
         * in the refined_vec.
         */
        template<typename T>
        void removeUnmarkedElements(
                const std::vector<T> &raw_vec,
                const std::vector<unsigned char> &markers,
                std::vector<T> &refined_vec) {
            if (raw_vec.size() != markers.size()) {
                ROS_WARN("The input size of raw_vec(%lu) and markers(%lu) does not match...", raw_vec.size(), markers.size());
            }
            for (int i = 0; i < markers.size(); ++i) {
                if (markers[i] == 0)
                    continue;
                refined_vec.push_back(raw_vec[i]);
            }
            return;
        }

        // Indicate if this is the first image message.
        bool is_first_img;

        // ID for the next new feature.
        FeatureIDType next_feature_id;

        // Feature detector
        ProcessorConfig processor_config;
        cv::Ptr<cv::Feature2D> detector_ptr;

        // IMU message buffer.
        std::vector<sensor_msgs::Imu> imu_msg_buffer;

        // Camera calibration parameters
        std::string cam0_distortion_model;
        cv::Vec2i cam0_resolution;
        cv::Vec4d cam0_intrinsics;
        cv::Vec4d cam0_distortion_coeffs;

        std::string cam1_distortion_model;
        cv::Vec2i cam1_resolution;
        cv::Vec4d cam1_intrinsics;
        cv::Vec4d cam1_distortion_coeffs;

        // Take a vector from cam0 frame to the IMU frame.
        cv::Matx33d R_cam0_imu;
        cv::Vec3d t_cam0_imu;
        // Take a vector from cam1 frame to the IMU frame.
        cv::Matx33d R_cam1_imu;
        cv::Vec3d t_cam1_imu;

        // Previous and current images
        cv_bridge::CvImageConstPtr cam0_prev_img_ptr;
        cv_bridge::CvImageConstPtr cam0_curr_img_ptr;
        cv_bridge::CvImageConstPtr cam1_curr_img_ptr;

        // Pyramids for previous and current image
        std::vector<cv::Mat> prev_cam0_pyramid_;
        std::vector<cv::Mat> curr_cam0_pyramid_;
        std::vector<cv::Mat> curr_cam1_pyramid_;

        // Features in the previous and current image.
        std::shared_ptr<GridFeatures> prev_features_ptr;
        std::shared_ptr<GridFeatures> curr_features_ptr;

        // Number of features after each outlier removal step.
        int before_tracking;    /// Number of the features before cam0 tracking.
        int after_tracking;     /// Number of features left after tracking.
        int after_matching;     /// Number of features left after stereo matching.
        int after_ransac;       /// Number of features after two Points ransac.

        // Ros node handle
        ros::NodeHandle nh;

        // Subscribers and publishers.
        message_filters::Subscriber<sensor_msgs::Image> cam0_img_sub;
        message_filters::Subscriber<sensor_msgs::Image> cam1_img_sub;
        message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> stereo_sub;
        ros::Subscriber imu_sub;
        ros::Publisher feature_pub;
        ros::Publisher tracking_info_pub;
        image_transport::Publisher debug_stereo_pub;

#if WITH_LIFETIME_STATISTICS
        // Debugging
        std::map<FeatureIDType, int> feature_lifetime;
        void updateFeatureLifetime();
        void featureLifetimeStatistics();
#endif        
    };

    typedef ImageProcessor::Ptr ImageProcessorPtr;
    typedef ImageProcessor::ConstPtr ImageProcessorConstPtr;

} // end namespace msckf_vio

#endif
