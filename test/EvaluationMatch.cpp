/***************************************
 @Time : 2020/7/10 上午10:07
 @Author : WenkyJong
 @Site : MianYang SWUST
 @File : EvaluationMatch.cpp
 @Contact: wenkyjong1996@gmail.com
 @desc:
*******/

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "AgainstNuclearCorner.h"

using namespace cv;
using std::cout;
using std::endl;

double EvaluateByHomogray(const Mat &homographyMatrix, const std::vector<DMatch> matchPoint, const std::vector<KeyPoint>keyPointNo1,std::vector<KeyPoint>keyPointNo2);
void DescriptorMatchMy(const Mat DescriptorNO1, const Mat DescriptorNo2,std::vector<DMatch>&matches);
bool FilterCorner(std::vector<KeyPoint>keyPoint,const int setNumber);

int main(int argc,char **argv) {

    Mat imageNo1 = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage/imgNoise1.bmp",
                          1);///home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage/imgNoise1.bmp
    Mat imageNo2 = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/NoiseImage/imgNoise2.bmp", 1);
    Mat imageGary1, imageGary2;
    cvtColor(imageNo1, imageGary1, CV_BGR2GRAY);
    cvtColor(imageNo2, imageGary2, CV_BGR2GRAY);

    //get the picture
    if (imageGary1.empty() || imageGary2.empty()) {
        cout << "can not load the picture" << endl;
    }
    imshow("the first image", imageGary1);
    imshow("the second image", imageGary2);


    //open file of homography matrix
    std::fstream fileHomography;
    fileHomography.open("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/H1to2p",
                        std::ios_base::in);//open file with read mode
    if (!fileHomography.is_open()) {
        cout << "can not open file" << endl;
        return -1;
    }

    Mat homography1to2 = Mat::zeros(3, 3, CV_32FC1);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            fileHomography >> homography1to2.at<float>(i, j);
        }
    }

    //1.using fast feature algorithm
    int setFastNumber{100};
    std::vector<KeyPoint> fastKeyPoints1;
    std::vector<KeyPoint> fastKeyPoints2;//
    Ptr<FastFeatureDetector> ptrFAST = FastFeatureDetector::create(50);

    ptrFAST->detect(imageGary1,fastKeyPoints1);
    ptrFAST->detect(imageGary2,fastKeyPoints2);
    //提取较强的点


    if(FilterCorner(fastKeyPoints1,50)){
        cout<<"the fast corner number is "<<50<<endl;
    }else{
        cout<<"the fast corner number is "<<fastKeyPoints1.size()<<endl;
    }

    cout<<"the fast corner number is "<<fastKeyPoints1.size()<<endl;
    //get the fast corner's descriptor using brief
    Mat fastDescriptors1;
    Mat fastDescriptors2;
    Ptr<DescriptorExtractor> featureBrief = xfeatures2d::BriefDescriptorExtractor::create();
    featureBrief->compute(imageGary1,fastKeyPoints1,fastDescriptors1);
    featureBrief->compute(imageGary2,fastKeyPoints2,fastDescriptors2);
    cout<<"the FAST descriptor size"<<endl;
    cout<<fastDescriptors1.size<<endl;

    //2.using Against Nuclear corner
    std::vector<KeyPoint> ANCKeyPointsNO1;
    std::vector<KeyPoint> ANCKeyPointsNO2;
    AgainstNuclearCorner ANCDetector1(100);//setting number of corners
    ANCDetector1.CalculateCorner(imageNo1);
    ANCKeyPointsNO1=ANCDetector1.getKeyPiont();
    AgainstNuclearCorner ANCDetector2(100);
    ANCDetector2.CalculateCorner(imageNo2);
    ANCKeyPointsNO2=ANCDetector2.getKeyPiont();

    Mat ANCDescriptorNO1;
    Mat ANCDescriptorNO2;
    Ptr<DescriptorExtractor> ANCDescriptor=xfeatures2d::BriefDescriptorExtractor::create();
    ANCDescriptor->compute(imageGary1,ANCKeyPointsNO1,ANCDescriptorNO1);
    ANCDescriptor->compute(imageGary2,ANCKeyPointsNO2,ANCDescriptorNO2);

    cout<<"the numbner of ANC corner "<<endl;
    cout<<ANCKeyPointsNO1.size()<<endl;
    cout<<"the ANC descriptor size"<<endl;
    cout<<ANCDescriptorNO1.size<<endl;

    //match evaluation fast corner
    std::vector<DMatch> matches;
    DescriptorMatchMy(fastDescriptors1,fastDescriptors2,matches);
    Mat matchImage;

    drawMatches(imageGary1,fastKeyPoints1,imageGary2,fastKeyPoints2,matches,matchImage,Scalar(255,255,0),Scalar(0,255,255));
    imshow("match resualt of fast",matchImage);

    double errorFast{1.0};
    errorFast =EvaluateByHomogray(homography1to2,matches,fastKeyPoints1,fastKeyPoints2);
    cout<<"all errors of fast feature is "<<errorFast<<endl;

    //match ANC
    std::vector<DMatch> matchsANC;//match ANC
    DescriptorMatchMy(ANCDescriptorNO1,ANCDescriptorNO2,matchsANC);
    Mat matchImageANC;
   // drawMatches(imageGary1,ANCDescriptorNO1,imageGary2,ANCDescriptorNO2,matchsANC,matchImageANC,Scalar(255,255,0),Scalar(0,255,255));

    double errorANC{0};
    errorANC=EvaluateByHomogray(homography1to2,matchsANC,ANCKeyPointsNO1,ANCKeyPointsNO2);
    cout<<"all errors of ANC feature is "<<errorANC<<endl;

    waitKey(0);
    getchar();
    return 0;

}



/*
 * chose the strongest response corner
 */
bool FilterCorner(std::vector<KeyPoint>keyPoint,const int setNumber){
    int cornerNumber=keyPoint.size();
    if(cornerNumber>setNumber){
        std::nth_element(keyPoint.begin(),keyPoint.begin()+setNumber,keyPoint.end(),[](KeyPoint& a,KeyPoint& b){return a.response>b.response;});
        keyPoint.erase(keyPoint.begin()+setNumber,keyPoint.end());
        return true;
    }else{
        return false;
    }

}

/*
 * the match with BruteForce
 */
void DescriptorMatchMy(const Mat DescriptorNO1, const Mat DescriptorNo2,std::vector<DMatch>&matches){

    int x;
    for(int i1=0;i1<DescriptorNO1.cols;i1++){//每一列表示一个特征点
        cv::DMatch m{i1,0,256};
        for(int i2=0;i2<DescriptorNo2.cols;i2++){
            int distance=0;
            for(int k=0;k<64;k++) {//每一位进行比较,描述子的长度为64x8
                x = DescriptorNO1.at<uchar>(k, i1) ^ DescriptorNo2.at<uchar>(k, i2);    //亦或的结果是以10进制保存需要转化二进制
                while (x){                                                              //采取将数字与其减一后的数据相与的结果（最后一位1消失）的次数判断1的个数
                    distance+=1;
                    x=x&(x-1);
                }
            }
            if(distance<m.distance){
                m.distance=distance;
                m.trainIdx=i2;
            }
        }
        matches.push_back(m);
    }
}

/*
 * using homographer to evalute the match
 */
double EvaluateByHomogray(const Mat &homographyMatrix, const std::vector<DMatch> matchPoint, const std::vector<KeyPoint>keyPointNo1,std::vector<KeyPoint>keyPointNo2){
    std::vector<Point2f> pointNo1;
    std::vector<Point2f> pointNo2;

    double errors{0};
    for(auto i:matchPoint){
        int indexQuery = i.queryIdx;
        int indexTrain = i.trainIdx;
        pointNo1.push_back(keyPointNo1[indexQuery].pt);
        pointNo2.push_back(keyPointNo2[indexTrain].pt);
    }

    int numberPoint=pointNo1.size();
    for(int i=0;i<numberPoint;i++){
        double s=pointNo1[i].x*homographyMatrix.at<float>(2,0)+pointNo1[i].y*homographyMatrix.at<float>(2,1)+homographyMatrix.at<float>(2,2);
        double projectX = (pointNo1[i].x*homographyMatrix.at<float>(0,0)+pointNo1[i].y*homographyMatrix.at<float>(0,1)+homographyMatrix.at<float>(0,2))/s;
        double projectY = (pointNo1[i].x*homographyMatrix.at<float>(1,0)+pointNo1[i].y*homographyMatrix.at<float>(1,1)+homographyMatrix.at<float>(1,2))/s;
        float errorsX=abs(pointNo2[i].x-projectX);
        float errorsY=abs(pointNo2[i].y-projectY);
        errors=errors+errorsX+errorsY;
    }
    int cntFeatures=pointNo1.size();
    errors=errors/cntFeatures;
    return errors;
}