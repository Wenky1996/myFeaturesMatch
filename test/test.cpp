/***************************************
 @Time : 2020/6/4 下午6:57
 @Author : WenkyJong
 @Site : MianYang SWUST
 @File : test.cpp
 @Contact: wenkyjong1996@gmail.com
 @desc:
*******/
#include <opencv2/opencv.hpp>
#include "AgainstNuclearCorner.h"

using namespace cv;
using std::cout;
using std::endl;

int main(int argc ,char **argv){

    Mat srcImage = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage/imgNoise6.bmp");
    if(srcImage.empty()){
        cout<<"can not load image"<<endl;
        return -1;
    }
    imshow("the original image",srcImage);

    AgainstNuclearCorner myConnerDetector(100);
    std::vector<Point2i>noiseLocation;
    myConnerDetector.FindNoise(srcImage);
    myConnerDetector.FastFeature(srcImage);
    myConnerDetector.CornerFilter();
    std::vector<KeyPoint> myCorner;
    myCorner=myConnerDetector.getKeyPiont();
    Mat dstImage;
    drawKeypoints(srcImage,myCorner,dstImage,Scalar(0,0,255),DrawMatchesFlags::DEFAULT);
    imshow("the corner image ",dstImage);

    //detect fast corner
    std::vector<KeyPoint> fastKeyPoint;
    Ptr<FastFeatureDetector> ptrFast=FastFeatureDetector::create(80);
    ptrFast->detect(srcImage,fastKeyPoint);
    Mat fastImage;
    drawKeypoints(srcImage,fastKeyPoint,fastImage,Scalar(255,0,0),DrawMatchesFlags::DEFAULT);
    imshow("the fast corner",fastImage);


    waitKey(0);
}

