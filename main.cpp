#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using std::cout;
using std::endl;


int main(int argc,char **argv) {
    Mat imageGary1 = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/img1.bmp",0);
    Mat imageGary2 = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/img2.bmp",0);

    if(imageGary1.empty()||imageGary2.empty()){
       cout<<"can not load the picture"<<endl;
    }
    imshow("the first image",imageGary1);
    imshow("the second image",imageGary2);

    std::vector<KeyPoint> keyPoints1;
    std::vector<KeyPoint> keyPoints2;
    Ptr<FastFeatureDetector> ptrFAST = FastFeatureDetector::create(40);

    ptrFAST->detect(imageGary1,keyPoints1);
    ptrFAST->detect(imageGary2,keyPoints2);


    Mat descriptors1;
    Mat descriptors2;
    Ptr<DescriptorExtractor> featureFREAK = xfeatures2d::FREAK::create();
    featureFREAK->compute(imageGary1,keyPoints1,descriptors1);
    featureFREAK->compute(imageGary2,keyPoints2,descriptors2);

    BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(descriptors1,descriptors2,matches);

    Mat matchImage;

    std::nth_element(matches.begin(),matches.begin()+100,matches.end());
    matches.erase(matches.begin()+100,matches.end());

    drawMatches(imageGary1,keyPoints1,imageGary2,keyPoints2,matches,matchImage,Scalar(255,255,0),Scalar(0,255,255));

    imshow("match resualt",matchImage);


    waitKey(0);
    getchar();
    return 0;
}
