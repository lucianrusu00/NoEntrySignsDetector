/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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

vector<Rect> getViolaFaces(Mat frame);

double getF1Score(vector<Rect> truth, vector<Rect> detected);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ){
    // 1. Read Input Image

    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    int imageIndex = getImageIndex(argv[1]);

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    }

    // 3. Detect Faces and Display Result
    vector<Rect> violaFaces = getViolaFaces(frame);
    showRectangles(violaFaces, frame, Scalar(0, 255, 0));
    std::cout << "Number of Viola faces: " << violaFaces.size() << '\n';
    for(int i = 0; i < violaFaces.size(); i++) std::cout << violaFaces[i] << ' ';
    std::cout << '\n';

    vector<Rect> groundTruthFaces = getGroundTruthFaces(imageIndex);
    std::cout << "Number of Truth faces: " << groundTruthFaces.size() << '\n';
    for(int i = 0; i < groundTruthFaces.size(); i++) std::cout << groundTruthFaces[i] << ' ';
    std::cout << '\n';
    showRectangles(groundTruthFaces, frame, Scalar(0,0,255));

    std::cout << "\nF1 score: " << getF1Score(groundTruthFaces, violaFaces) << '\n';

    // 4. Save Result Image
    char *fileToOutput;
    sprintf(fileToOutput, "allDetected/detected%d.jpg", imageIndex);
    imwrite(fileToOutput, frame);

	return 0;
}

int getImageIndex(string imageName){
    string number = "";

    for(int i = 0; i < imageName.size(); i++){
        if(isnumber(imageName[i]))
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

    ifstream readFromFile("GroundTruth.csv");

    string read;

    vector<Rect> groundTruthRects;

    while(getline (readFromFile, read)){
        if( isnumber(read[0]) && atoi(read.c_str()) == imageIndex ) {
            groundTruthRects = getRectangleFaces(readFromFile);
            break;
        }
    }

    return groundTruthRects;

}

/** @function detectAndDisplay */
vector<Rect> getViolaFaces( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

    return faces;

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