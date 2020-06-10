/**************************************
 @Time : 2020/6/1 下午2:58
 @Author : WenkyJong
 @Site : MianYang SWUST
 @File : AgainstNuclearCorner.cpp
 @Contact: wenkyjong1996@gmail.com
 @desc:
 ********/


#ifndef FEATUREMATCHEVALUATION_AGAINSTNUCLEARCORNER_H
#define FEATUREMATCHEVALUATION_AGAINSTNUCLEARCORNER_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>

using namespace cv;

class AgainstNuclearCorner{
public:
    AgainstNuclearCorner(const int &numberCorners);
    inline int getNoiseNumber(){ return _noiseNumber;} //get noise number
    void CalculateCorner(Mat& srcImage);
    std::vector<KeyPoint>getKeyPiont(){ return keyPoints;}
private:
    int _numberConner{0};
    int _noiseNumber{0};
    std::vector<Point2i> noiseLocation;
    std::vector<KeyPoint> keyPointFast;
    std::vector<KeyPoint> keyPointFilte;
    std::vector<KeyPoint> keyPoints;
    void FindNoise(Mat &noiseImage);
    void FastFeature(Mat &noiseImage);
    void CornerFilter();
    //void FindNoise(Mat &noiseImage,std::vector<Point2i> &noiseLocation);

};


#endif //FEATUREMATCHEVALUATION_AGAINSTNUCLEARCORNER_H

