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
#include <nmmintrin.h>

using namespace cv;
using std::cout;
using std::endl;

void DescriptorMatchMy(const Mat DescriptorNO1, const Mat DescriptorNo2,std::vector<DMatch>&maches);

int main(int argc ,char **argv){


    /*
     * test XOR operation with bit
     */
    Mat data1{3,3,CV_8U,Scalar{8}};
    cout<<data1<<endl;
    Mat data2{3,3,CV_8U,Scalar {2}};
    cout<<data2<<endl;
    int distance=0;
    int x=0;
    int y=0;
    for(int i=0;i<data2.cols;i++){
        for(int j=0;j<data2.rows;j++) {
            x = data2.at<uchar>(i, j) ^ data1.at<uchar>(i, j);
            while (x != 0) {
                distance += 1;
                x = x & (x - 1);
            }
        }
    }
    cout<<distance<<endl;


    Mat srcImage = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage/imgNoise1.bmp");

    if(srcImage.empty()){
        cout<<"can not load image"<<endl;
        return -1;
    }
    imshow("the original image",srcImage);


    AgainstNuclearCorner myConnerDetector(100);
    std::vector<Point2i>noiseLocation;
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

void DescriptorMatchMy(const Mat DescriptorNO1, const Mat DescriptorNo2,std::vector<DMatch>&matches){

    int d_max{40};
    for(int i1=0;i1<DescriptorNO1.cols;i1++){//每一列表示一个特征点
        cv::DMatch m{i1,0,256};
        for(int i2=0;i2<DescriptorNo2.cols;i2++){
            int distance=1;
            for(int k=0;k<64;k++)//每一位进行比较,描述子的长度为64x8
                //distance+=_mm_popcnt_u32(DescriptorNO1.at<uchar>(k,i1)^DescriptorNo2.at<uchar>(k,i2));
            if(distance<d_max&&distance<m.distance){
                m.distance=distance;
                m.trainIdx=i2;
            }
        }
        if(m.distance<d_max){
            matches.push_back(m);
        }
    }
}

