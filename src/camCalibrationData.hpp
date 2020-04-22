#ifndef CAMERA_CALIBRATION_DATA_HPP
#define CAMERA_CALIBRATION_DATA_HPP

#include <opencv2/core.hpp>

void loadCalibrationData(cv::Mat &P_rect_00, cv::Mat &R_rect_00, cv::Mat &RT);

#endif // CAMERA_CALIBRATION_DATA_HPP
