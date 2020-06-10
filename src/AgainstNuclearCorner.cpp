/***************************************
 @Time : 2020/6/2 下午4:49
 @Author : WenkyJong
 @Site : MianYang SWUST
 @File : AgainstNuclearCorner.cpp
 @Contact: wenkyjong1996@gmail.com
 @desc:
*******/

#include "AgainstNuclearCorner.h"

AgainstNuclearCorner::AgainstNuclearCorner(const int &numberCorners):_numberConner(numberCorners){

}

/**
 * get the location of noise format (x,y)
 * @param noiseImage
 * @param noiseLocation
 */
void AgainstNuclearCorner::FindNoise(Mat &noiseImage){
    std::cout<<"hello"<<std::endl;
    Mat channel[3];
    split(noiseImage,channel);
    Mat_<uchar > channelB{channel[0]};
    Mat_<uchar > channelG{channel[1]};
    Mat_<uchar > channelR{channel[2]};
    int rows{noiseImage.rows};
    int cols{noiseImage.cols};
    _noiseNumber=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            int valueB{channelB(i,j)};
            int valueG{channelG(i,j)};
            int valueR{channelR(i,j)};
            int totalGary{valueB+valueG+valueR};
            float rateB{static_cast<float >(valueB)/totalGary};
            float rateG{static_cast<float >(valueG)/totalGary};
            float rateR{static_cast<float >(valueR)/totalGary};
            if(rateB>0.8||rateG>0.8||rateR>0.8){
                Point2i currLocation{i,j};
                noiseLocation.push_back(currLocation);
                _noiseNumber++;
            }
        }
    }

}

/**
 * get key points using fast detector
 * @param noiseImage
 */

void AgainstNuclearCorner::FastFeature(Mat &noiseImage) {
    Mat garyImage;
    cvtColor(noiseImage,garyImage,CV_BGR2GRAY);
    Ptr<FastFeatureDetector> fastPtr = FastFeatureDetector::create(40);
    fastPtr->detect(garyImage,keyPointFast);
}

/**
 * acording to distance to filte the fast corner
 */

void AgainstNuclearCorner::CornerFilter() {
    std::multimap<int,KeyPoint> mapDistanceCornor;//可以有重复的键值即距离
    int numberFast=keyPointFast.size();
    int numberNoise=noiseLocation.size();
    int miniDistance{10000};
    for(int i=0;i<numberFast;i++){
        int xPoint = keyPointFast[i].pt.x;
        int yPoint = keyPointFast[i].pt.y;
        miniDistance=10000;
        for(int j=0;j<numberNoise;j++){
            int xNoise = noiseLocation[j].x;
            int yNoise = noiseLocation[j].y;
            int distance=pow(xPoint-xNoise,2)+pow(yPoint-yNoise,2);
            if(distance<miniDistance){
                miniDistance=distance;
            }
        }
        mapDistanceCornor.insert(std::pair<int,KeyPoint>(miniDistance,keyPointFast[i]));//向map中插入数据
    }
    auto iter1 = mapDistanceCornor.upper_bound(8000);//返回键值即距离大于100的迭代器

    for(auto iter = iter1;iter!=mapDistanceCornor.end();iter++){
        keyPointFilte.push_back(iter->second);
    }

    //利用得分进行筛选
    int numberCornersFilte=keyPointFilte.size();
    if(numberCornersFilte>_numberConner){
        std::nth_element(keyPointFilte.begin(),keyPointFilte.begin()+_numberConner,keyPointFilte.end(),[](cv::KeyPoint& a,cv::KeyPoint& b){ return a.response>b.response;});
        keyPoints.resize(_numberConner);
        for(int i=0;i<_numberConner;i++){
            keyPoints.push_back(keyPointFilte[i]);
        }
    } else{
        for(auto& i:keyPointFilte){
            keyPoints.push_back(i);
        }
    }

}


void AgainstNuclearCorner::CalculateCorner(Mat& srcImage) {
    FindNoise(srcImage);
    FastFeature(srcImage);
    CornerFilter();
}