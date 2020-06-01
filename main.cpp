#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using std::cout;
using std::endl;


int main(int argc,char **argv) {
    Mat imageGary1 = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/img1.bmp",0);
    Mat imageGary2 = imread("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/img2.bmp",0);
    cout<<204*-2.3911512e-06+319*2.9032886e-06+5.7865196e-01<<endl;
    //open file of homography matrix
    std::fstream fileHomography;
    fileHomography.open("/home/zwk/CLionProjects/FeatureMatchEvaluation/bmp/H1to2p",std::ios_base::in);//open file with read mode
    if(!fileHomography.is_open()){
        cout<<"can not open file"<<endl;
        return -1;
    }

    Mat homography1to2=Mat::zeros(3,3,CV_32FC1);
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
           fileHomography>>homography1to2.at<float>(i,j);
        }
    }
    cout<<homography1to2<<endl;

    //get the picture
    if(imageGary1.empty()||imageGary2.empty()){
       cout<<"can not load the picture"<<endl;
    }
    imshow("the first image",imageGary1);
    imshow("the second image",imageGary2);

    //1.using fast feature algorithm
    std::vector<KeyPoint> fastKeyPoints1;
    std::vector<KeyPoint> fastKeyPoints2;
    Ptr<FastFeatureDetector> ptrFAST = FastFeatureDetector::create(40);

    ptrFAST->detect(imageGary1,fastKeyPoints1);
    ptrFAST->detect(imageGary2,fastKeyPoints2);
    Mat fastDescriptors1;
    Mat fastDescriptors2;
    Ptr<DescriptorExtractor> featureFREAK = xfeatures2d::FREAK::create();
    featureFREAK->compute(imageGary1,fastKeyPoints1,fastDescriptors1);
    featureFREAK->compute(imageGary2,fastKeyPoints2,fastDescriptors2);

    //2.using orb feature algorithm
    std::vector<KeyPoint> orbKeyPoint1;
    std::vector<KeyPoint> orbKeyPoint2;
    Ptr<ORB> ptrORB = ORB::create(40);
    ptrORB->detect(imageGary1,orbKeyPoint1);
    ptrORB->detect(imageGary2,orbKeyPoint2);
    Mat orbDescriptors1;
    Mat orbDescriptors2;
    ptrORB->compute(imageGary1,orbKeyPoint1,orbDescriptors1);
    ptrORB->compute(imageGary2,orbKeyPoint2,orbDescriptors2);

    //match the points of fast and draw the picture
    BFMatcher matcher(NORM_HAMMING);
    std::vector<DMatch> matches;
    matcher.match(fastDescriptors1,fastDescriptors2,matches);

    Mat matchImage;
    std::nth_element(matches.begin(),matches.begin()+20,matches.end());
    matches.erase(matches.begin()+20,matches.end());

    drawMatches(imageGary1,fastKeyPoints1,imageGary2,fastKeyPoints2,matches,matchImage,Scalar(255,255,0),Scalar(0,255,255));
    imshow("match resualt of fast",matchImage);

    //get the coordinate of match keyPoint of fast
    std::vector<Point2f> pointsFast1,pointsFast2;
    for(auto &i : matches){
        int idxQuery =i.queryIdx;
        int idexTrain=i.trainIdx;
        pointsFast1.push_back(fastKeyPoints1[idxQuery].pt);
        pointsFast2.push_back(fastKeyPoints2[idexTrain].pt);
    }

    int idxFast{0};
    float allErrorsFast{0};
    for(auto &i:pointsFast1){
        float s=i.x*homography1to2.at<float>(2,0)+i.y*homography1to2.at<float>(2,1)+homography1to2.at<float>(2,2);
        float xTrain=(i.x*homography1to2.at<float>(0,0)+i.y*homography1to2.at<float>(0,1)+homography1to2.at<float>(0,2))/s;
        float yTrain=(i.x*homography1to2.at<float>(1,0)+i.y*homography1to2.at<float>(1,1)+homography1to2.at<float>(1,2))/s;
        float xErrors=abs(pointsFast2[idxFast].x-xTrain);
        float yErrors=abs(pointsFast2[idxFast].y-yTrain);
        idxFast++;
        allErrorsFast=allErrorsFast+xErrors+yErrors;
    }
    cout<<"the fast feature all errors piexel is "<<allErrorsFast<<endl;


    //match orb feature with BF
    BFMatcher matcherOrb;
    std::vector<DMatch> matchesOrb;
    matcherOrb.match(orbDescriptors1,orbDescriptors2,matchesOrb);

    Mat matchImageOrb;
    std::nth_element(matchesOrb.begin(),matchesOrb.begin()+20,matchesOrb.end());
    matchesOrb.erase(matchesOrb.begin()+20,matchesOrb.end());
    drawMatches(imageGary1,orbKeyPoint1,imageGary2,orbKeyPoint2,matchesOrb,matchImageOrb,Scalar(0,255,255),Scalar(0,255,0));
    imshow("match result of orb",matchImageOrb);

    //get the coordinate of match keyPoint of orb and calculate the errors
    std::vector<Point2f> pointsOrb1,pointsOrb2;
    for(auto &i:matchesOrb){
        int idxQuery=i.queryIdx;
        int idxTrain=i.trainIdx;
        pointsOrb1.push_back(orbKeyPoint1[idxQuery].pt);
        pointsOrb2.push_back(orbKeyPoint2[idxTrain].pt);
    }
    Mat homographyMatrixOrb;
    std::vector<char> inliers;
    homographyMatrixOrb = findHomography(pointsFast1,pointsFast2,inliers,cv::RANSAC,1);
    cout<<homographyMatrixOrb<<endl;
    int idx{0};
    float allErrors{0};
    for(auto &i:pointsOrb1){
        float s=i.x*homography1to2.at<float>(2,0)+i.y*homography1to2.at<float>(2,1)+homography1to2.at<float>(2,2);
        float xTrain=(i.x*homography1to2.at<float>(0,0)+i.y*homography1to2.at<float>(0,1)+homography1to2.at<float>(0,2))/s;
        float yTrain=(i.x*homography1to2.at<float>(1,0)+i.y*homography1to2.at<float>(1,1)+homography1to2.at<float>(1,2))/s;
        float xError = abs(xTrain-pointsOrb2[idx].x);
        float yError = abs(yTrain-pointsOrb2[idx].y);
        idx++;
        allErrors=allErrors+yError+xError;
    }
    cout<<"the all errors of orb feature is "<<allErrors<<endl;

    waitKey(0);
    getchar();
    return 0;
}


