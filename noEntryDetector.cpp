#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <string.h>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
vector<Rect> getGroundTruthFaces(int imageIndex);

void showRectangles(vector<Rect> rectangles, Mat frame, Scalar color);

int getImageIndex(string imageName);

vector<Rect> detectSigns(Mat frame);

double getF1Score(vector<Rect> truth, vector<Rect> detected);

void calculateSobel(Mat grayImage, Mat &sobelX, Mat &sobelY, Mat &magnitudeImage, Mat &directionImage);

void threshold(Mat &grayImage, double thershold, double maxValue);

vector<Vec3f> getHoughCircles(Mat magnitudeImage, Mat directionImage, Mat image);

void filterNoEntrySigns(vector<Rect> &noEntrySigns, vector<Vec3f> houghCircles);



/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ){
    // 1. Read Input Image

    for(int j = 0; j <= 15; j++){
        char* fileToInput = new char[100];

    sprintf(fileToInput, "No_entry/NoEntry%d.bmp", j);


    //Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat frame = imread(fileToInput, CV_LOAD_IMAGE_COLOR);
    int imageIndex = j;
    //int imageIndex = getImageIndex(argv[1]);

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    }

    // 3. Detect Faces and Display Result
    vector<Rect> signs = detectSigns(frame);
    showRectangles(signs, frame, Scalar(0, 255, 0));
    std::cout << "Number of Viola NoEntry signs: " << signs.size() << '\n';
    for(int i = 0; i < signs.size(); i++) std::cout << signs[i] << ' ';
    std::cout << '\n';

    vector<Rect> groundTruthFaces = getGroundTruthFaces(imageIndex);
    std::cout << "Number of Truth NoEntry signs: " << groundTruthFaces.size() << '\n';
    for(int i = 0; i < groundTruthFaces.size(); i++) std::cout << groundTruthFaces[i] << ' ';
    std::cout << '\n';
    showRectangles(groundTruthFaces, frame, Scalar(0,0,255));

    std::cout << "\nF1 score: " << getF1Score(groundTruthFaces, signs) << '\n';

    // 4. Save Result Image
    char* fileToOutput = new char[100];

    sprintf(fileToOutput, "allDetected/detected%d.jpg", imageIndex);
    imwrite(fileToOutput, frame);

    delete(fileToInput);
    delete(fileToOutput);
    }


	return 0;
}

int getImageIndex(string imageName){
    string number = "";

    for(int i = 0; i < imageName.size(); i++){
        if(isdigit(imageName[i]))
            number.push_back(imageName[i]);
    }

    return atoi(number.c_str());
}

vector<Rect> getRectangleFaces(std::istream& is){
    vector<Rect> groundTruthRects;

    string line;

    while(getline(is, line)){
        if(line[0] == '-') return groundTruthRects;
        char* numbers = strtok(strdup(line.c_str()), ";");
        int posX = atoi(numbers);
        numbers = strtok(NULL, ";");
        int posY = atoi(numbers);
        numbers = strtok(NULL, ";");
        int width = atoi(numbers);
        numbers = strtok(NULL, ";");
        int height = atoi(numbers);
        groundTruthRects.push_back(Rect(posX,posY,width,height));
    }

    return groundTruthRects;
}

vector<Rect> getGroundTruthFaces(int imageIndex){

    ifstream readFromFile("GroundTruthNoEntry.csv");

    string read;

    vector<Rect> groundTruthRects;

    while(getline (readFromFile, read)){
        if( isdigit(read[0]) && atoi(read.c_str()) == imageIndex ) {
            groundTruthRects = getRectangleFaces(readFromFile);
            break;
        }
    }


    return groundTruthRects;

}

/** @function detectAndDisplay */
vector<Rect> detectSigns( Mat image )
{
	std::vector<Rect> noEntrySigns;
	Mat grayImage;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( image, grayImage, CV_BGR2GRAY );
	equalizeHist( grayImage, grayImage );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( grayImage, noEntrySigns, 1.03, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

    GaussianBlur(grayImage, noEntrySigns, Size(3,3), 0, 0, BORDER_DEFAULT);

    Mat sobelX, sobelY, magnitudeImage, directionImage;

    calculateSobel(grayImage, sobelX, sobelY, magnitudeImage, directionImage);

    threshold(grayImage, 120, 255);

    vector<Vec3f> houghCircles = getHoughCircles(magnitudeImage, directionImage, image);

    filterNoEntrySigns(noEntrySigns, houghCircles);




       // 3. Print number of Faces found
	std::cout << noEntrySigns.size() << std::endl;

    return noEntrySigns;

}

bool intersectionOverUnion(Rect one, Rect two){
    double areaIntersected = (one & two).area();
    double areaUnion = (one | two).area();

    double intersectionOverUnion = areaIntersected/areaUnion;
    if(intersectionOverUnion > 0.25) // more than 25%
        return true;

    return false;
}

double getTruePositive(vector<Rect> truth, vector<Rect> detected){
    double truePositive = 0;

    for(int i = 0; i < truth.size(); i++){
        for(int j = 0; j < detected.size(); i++){
            if(intersectionOverUnion(truth[i], detected[j])){
                truePositive++;
                j = detected.size();
                continue;
            }
        }

    }

    return truePositive;
}


double getF1Score(vector<Rect> truth, vector<Rect> detected){
    double truePositive = getTruePositive(truth, detected);
    double falsePositive = detected.size() - truePositive;
    double falseNegative = truth.size() - truePositive;

    return truePositive / (truePositive + (falsePositive + falseNegative)/2);
}

void showRectangles(vector<Rect> rectangles, Mat frame, Scalar color){

    for( int i = 0; i < rectangles.size(); i++ )
    {
        rectangle(frame, Point(rectangles[i].x, rectangles[i].y), Point(rectangles[i].x + rectangles[i].width, rectangles[i].y + rectangles[i].height), color, 2);
    }
}

vector<Vec3f> getHoughCircles(Mat magnitudeImage, Mat directionImage, Mat image){
    return vector<Vec3f>();
}