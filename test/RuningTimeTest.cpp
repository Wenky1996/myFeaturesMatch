/***************************************
 @Time : 2020/7/2 上午10:28
 @Author : WenkyJong
 @Site : MianYang SWUST
 @File : RuningTimeTest.cpp
 @Contact: wenkyjong1996@gmail.com
 @desc:
*******/
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "AgainstNuclearCorner.h"

using namespace cv;
int main(int argc,char** argv){

   // Mat image1=imread("~/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage21/imgNoise1.bmp");
    Mat image1=imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage21/imgNoise1.bmp");
    Mat image2=imread("~/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage21/imgNoise2.bmp");

    if(image1.empty()){
        std::cout<<"can not load the picture"<<std::endl;
    }
    std::cout<<image1.size<<std::endl;
    //the ANC corner
    AgainstNuclearCorner ANCer(10);
    std::vector<KeyPoint> ANCkeyPont;
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    ANCer.CalculateCorner(image1);
    std::chrono::steady_clock::time_point t2=std::chrono::steady_clock::now();
    std::chrono::duration<double >time_used=std::chrono::duration_cast<std::chrono::duration<double >>(t2-t1);
    std::cout<<"the using time of ANC corner is "<<time_used.count()<<std::endl;

    //the Harris corner
   // Mat harrisCorner;
   // cornerHarris(image1,harrisCorner,3,3,0.01);

    //the FAST corner
    cvtColor(image1,image1,CV_BGR2GRAY);
    std::vector<KeyPoint> fastKeyPoint;
    t1=std::chrono::steady_clock::now();
    Ptr<FastFeatureDetector> ptrFast=FastFeatureDetector::create(50);
    ptrFast->detect(image1,fastKeyPoint);
    t2=std::chrono::steady_clock::now();
    time_used=std::chrono::duration_cast<std::chrono::duration<double >>(t2-t1);
    std::cout<<"the using time of FAST is "<<time_used.count()<<std::endl;

    //the sift corner
    std::vector<KeyPoint> siftkeyPoint;
    t1=std::chrono::steady_clock::now();
    Ptr<cv::xfeatures2d::SiftFeatureDetector>ptrSIFT=xfeatures2d::SiftFeatureDetector::create(100);
    ptrSIFT->detect(image1,siftkeyPoint);
    t2=std::chrono::steady_clock::now();
    time_used=std::chrono::duration_cast<std::chrono::duration<double >>(t2-t1);
    std::cout<<"the using time of SITF is "<<time_used.count()<<std::endl;


    return 0;
}
