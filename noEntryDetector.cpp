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

void threshold(Mat &grayImage, double threshold, double maxValue);

vector<Vec3f> getHoughCircles(Mat magnitudeImage, Mat directionImage, Mat image, int maximumCircleRadius = 110);

void filterNoEntrySigns(vector<Rect> &noEntrySigns, vector<Vec3f> houghCircles);



/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ){
    // 1. Read Input Image

    //char* fileToInput = new char[100];

    //sprintf(fileToInput, "No_entry/NoEntry%d.bmp", j);


    Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    //Mat frame = imread(fileToInput, CV_LOAD_IMAGE_COLOR);
    //int imageIndex = j;
    int imageIndex = getImageIndex(argv[1]);

    // 2. Load the Strong Classifier in a structure called `Cascade'
    if (!cascade.load(cascade_name)) {
        printf("--(!)Error loading\n");
        return -1;
    }

    // 3. Detect Faces and Display Result
    vector<Rect> signs = detectSigns(frame);
    showRectangles(signs, frame, Scalar(0, 255, 0));
    //std::cout << "Number of Viola NoEntry signs: " << signs.size() << '\n';
    //for(int i = 0; i < signs.size(); i++) std::cout << signs[i] << ' ';
    //std::cout << '\n';

    vector<Rect> groundTruthFaces = getGroundTruthFaces(imageIndex);
    //std::cout << "Number of Truth NoEntry signs: " << groundTruthFaces.size() << '\n';
    //for(int i = 0; i < groundTruthFaces.size(); i++) std::cout << groundTruthFaces[i] << ' ';
    //std::cout << '\n';
    showRectangles(groundTruthFaces, frame, Scalar(0,0,255));

    std::cout << "\nF1 score for image "<< imageIndex << " is " << getF1Score(groundTruthFaces, signs) << '\n';

    // 4. Save Result Image
    char* fileToOutput = new char[100];

    sprintf(fileToOutput, "allDetected/detected%d.jpg", imageIndex);
    imwrite(fileToOutput, frame);

    //delete[] (fileToInput);
    delete[] (fileToOutput);


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

	cvtColor( image, grayImage, CV_BGR2GRAY );
	equalizeHist( grayImage, grayImage );

	cascade.detectMultiScale( grayImage, noEntrySigns, 1.03, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );


    GaussianBlur(grayImage, grayImage, Size(3,3), 0, 0, BORDER_DEFAULT);

    Mat sobelX, sobelY, magnitudeImage, directionImage;

    sobelX.create(grayImage.rows, grayImage.cols, CV_64F);
    sobelY.create(grayImage.rows, grayImage.cols, CV_64F);
    calculateSobel(grayImage, sobelX, sobelY, magnitudeImage, directionImage);

    threshold(magnitudeImage, 128, 255);
    imwrite("magnitude15.jpg", magnitudeImage);


    vector<Vec3f> houghCircles = getHoughCircles(magnitudeImage, directionImage, image);

    filterNoEntrySigns(noEntrySigns, houghCircles);

	std::cout << "Number of signs found: " << noEntrySigns.size();

    return noEntrySigns;

}

bool intersectionOverUnion(Rect one, Rect two){
    double areaIntersected = (one & two).area();
    double areaUnion = (one | two).area();

    double intersectionOverUnion = areaIntersected/areaUnion;
    if(intersectionOverUnion > 0.25)
        return true;

    return false;
}

double getTruePositive(vector<Rect> truth, vector<Rect> detected){
    double truePositive = 0;

    for(int i = 0; i < truth.size(); i++){
        for(int j = 0; j < detected.size(); j++){
            if(intersectionOverUnion(truth[i], detected[j])){
                truePositive++;
                j = detected.size();
            }
        }

    }

    return truePositive;
}


double getF1Score(vector<Rect> truth, vector<Rect> detected){
    double truePositive = getTruePositive(truth, detected);
    double falsePositive = detected.size() - truePositive;
    double falseNegative = truth.size() - truePositive;

    //std::cout << "True positive rate: " << truePositive/truth.size() << '\n';

    return truePositive / (truePositive + (falsePositive + falseNegative)/2.0);
}

void showRectangles(vector<Rect> rectangles, Mat frame, Scalar color){

    for( int i = 0; i < rectangles.size(); i++ )
    {
        rectangle(frame, Point(rectangles[i].x, rectangles[i].y), Point(rectangles[i].x + rectangles[i].width, rectangles[i].y + rectangles[i].height), color, 2);
    }
}

Mat createEmptyKernel(int kernelSize){

    if(kernelSize < 0 || kernelSize % 2 == 0){
        printf("The kernel size should be an odd, positive number. Returned empty kernel!\n");
        return Mat();
    }

    cv::Mat kX;
    cv::Mat kY;

    kX.create(cv::Size(1, kernelSize), CV_64F);
    kY.create(cv::Size(kernelSize, 1), CV_64F);

    cv::Mat kernel = kX * kY;

    return kernel;
}

void threshold(Mat &grayImage, double threshold, double maxValue){
    for(int x = 0; x < grayImage.cols; x++)
        for(int y = 0; y < grayImage.rows; y++)
            if(grayImage.at<double>(y,x) >= threshold) grayImage.at<double>(y,x) = maxValue;
            else grayImage.at<double>(y,x) = 0;
}

void computeGradientMagnitude(Mat grayImage, Mat sobelX, Mat sobelY, Mat &magnitudeImage){
    for (int y = 0; y < grayImage.rows; y++) {
        for (int x = 0; x < grayImage.cols; x++) {
            magnitudeImage.at<double>(y,x) = sqrt(sobelX.at<double>(y,x) * sobelX.at<double>(y,x) + sobelY.at<double>(y,x) * sobelY.at<double>(y,x));
        }
    }
}

void computeGradientDirection(Mat grayImage, Mat sobelX, Mat sobelY, Mat &directionImage){
    for (int y = 0; y < grayImage.rows; y++) {
        for (int x = 0; x < grayImage.cols; x++) {
            directionImage.at<double>(y,x) = atan2(sobelY.at<double>(y, x), sobelX.at<double>(y, x));
        }
    }
}

void computeSobelX(Mat grayImage, Mat &sobelX) {

    Mat kernelX = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

    Mat kernelY = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    int kernelRadiusX = ( kernelX.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernelY.size[1] - 1 ) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder( grayImage, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,cv::BORDER_REPLICATE );

    for (int i = 0; i < grayImage.rows; i++) {
        for (int j = 0; j < grayImage.cols; j++) {
            double sumX = 0;
            for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
                for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {

                    int posImageX = i + m + kernelRadiusX;
                    int posImageY = j + n + kernelRadiusY;
                    int posKernelX = m + kernelRadiusX;
                    int posKernelY = n + kernelRadiusY;

                    double kernelDxVal = kernelX.at<double>(posKernelX, posKernelY);

                    sumX += paddedInput.at<uchar>(posImageX, posImageY) * kernelDxVal;

                }
            }

            sobelX.at<double>(i, j) = sumX;
        }
    }
}

void computeSobelY(Mat grayImage, Mat &sobelY) {

    Mat kernelX = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);

    Mat kernelY = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    int kernelRadiusX = ( kernelX.size[0] - 1 ) / 2;
    int kernelRadiusY = ( kernelY.size[1] - 1 ) / 2;

    cv::Mat paddedInput;
    cv::copyMakeBorder( grayImage, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,cv::BORDER_REPLICATE );

    for (int i = 0; i < grayImage.rows; i++) {
        for (int j = 0; j < grayImage.cols; j++) {
            double sumY = 0;
            for (int m = -kernelRadiusX; m <= kernelRadiusX; m++) {
                for (int n = -kernelRadiusY; n <= kernelRadiusY; n++) {

                    int posImageX = i + m + kernelRadiusX;
                    int posImageY = j + n + kernelRadiusY;
                    int posKernelX = m + kernelRadiusX;
                    int posKernelY = n + kernelRadiusY;

                    double kernelDyVal = kernelY.at<double>(posKernelX, posKernelY);

                    sumY += paddedInput.at<uchar>(posImageX, posImageY) * kernelDyVal;

                }
            }

            sobelY.at<double>(i, j) = sumY;
        }
    }
}

void calculateSobel(Mat grayImage, Mat &sobelX, Mat &sobelY, Mat &magnitudeImage, Mat &directionImage){
    magnitudeImage.create(grayImage.rows, grayImage.cols, CV_64F);
    directionImage.create(grayImage.rows, grayImage.cols, CV_64F);
    computeSobelX(grayImage, sobelX);
    computeSobelY(grayImage, sobelY);
    computeGradientMagnitude(grayImage, sobelX, sobelY, magnitudeImage);
    computeGradientDirection(grayImage, sobelX, sobelY, directionImage);
}

void computeHough3D(Mat image,Mat magnitudeImage, Mat directionImage, Mat &houghSpace, int maximumCircleRadius){

    for(int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (magnitudeImage.at<double>(y, x) > 0) {
                for (int r = 0; r < maximumCircleRadius; r++)
                    for (int i = -1; i <= 1; i += 2) {
                        for (int j = -1; j <= 1; j += 2) {
                            int xActual = x + i * r * cos(directionImage.at<double>(y, x));
                            int yActual = y + j * r * sin(directionImage.at<double>(y, x));
                            if (xActual >= 0 && xActual < image.cols && yActual >= 0 && yActual < image.rows) {
                                houghSpace.at<double>(yActual, xActual, r)++;
                            }
                        }
                    }
            }
        }
    }

}

void computeSummedRadius(Mat image, Mat houghSpace, Mat &summedRadius, int maximumCircleRadius){

    for (int r = 0; r < maximumCircleRadius; r++) {
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {

                if (r > 10 && houghSpace.at<double>(y, x, r) >= 15 ){
                    houghSpace.at<double>(y, x, r) = 255.0;
                }
                else{
                    houghSpace.at<double>(y, x, r) = 0;
                }

                summedRadius.at<double>(y, x) += houghSpace.at<double>(y, x, r);
            }
        }
    }


}

void getBestRadius(Mat image, Mat summedRadius, Mat &radBest, Mat houghSpace, int maximumCircleRadius){

    for (int r = 0; r < maximumCircleRadius; r++) {
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                if (r > 10 && houghSpace.at<double>(y, x, r) >= 15 ){
                    radBest.at<double>(y,x) = max( (int) radBest.at<double>(y,x), r);
                }
            }
        }
    }

}

bool distanced(vector<Vec3f> possibleSolutions, int x, int y){

    for(int i = 0; i < possibleSolutions.size(); i++){
        if(abs(x - possibleSolutions[i][0]) < 20 && abs(y - possibleSolutions[i][1]) < 20) {
            return false;
        }
    }

    return true;
}

vector<Vec3f> getPossibleSolutions(Mat image, Mat summedRadius, Mat houghSpace, int maximumCircleRadius){

    vector<Vec3f> possibleSolutions;
    Mat bestRadius(2, summedRadius.size, CV_64F, Scalar::all(0));
    getBestRadius(image, summedRadius, bestRadius, houghSpace, maximumCircleRadius);


    for (int y = 1; y < summedRadius.rows - 1; y++) {
        for (int x = 1; x < summedRadius.cols - 1; x++) {


            int cluster = summedRadius.at<double>(y , x);;

            for(int i = -1; i <= 1; i += 2){
                cluster += summedRadius.at<double>(y + i, x);
                cluster += summedRadius.at<double>(y, x + i);
            }
            if (bestRadius.at<double>(y,x) > 10 && cluster >= 3 * 255.0 && distanced(possibleSolutions, x, y)){
                possibleSolutions.push_back(Vec3f(x, y, bestRadius.at<double>(y,x)));
            }

        }
    }

    return possibleSolutions;
}

void deleteNeighbouringCircles(Mat &summedRadius, int y, int x, int deleteRadius) {

    for(int yNew = y - deleteRadius; yNew <= y + deleteRadius; yNew++){
        for(int xNew = x - deleteRadius; xNew <= x + deleteRadius; xNew++){
            if(yNew >= 0 && yNew < summedRadius.rows)
                if(xNew >= 0 && xNew < summedRadius.cols)
                    summedRadius.at<double>(yNew, xNew) = 0;
        }
    }

}


vector<Vec3f> selectMostLikelySolutions(vector<Vec3f> possibleSolutions, Mat houghSpace, Mat summedRadius, int maximumCircleRadius){
    vector<Vec3f> solution;

    for (int i = 0; i < possibleSolutions.size(); i++) {
        int max = -1;
        int bestRadius = -1;
        int bestX = -1;
        int bestY = -1;

        for (int radius = 0; radius < maximumCircleRadius; radius++) {

            if (houghSpace.at<double>(possibleSolutions[i][1], possibleSolutions[i][0], radius) > max && summedRadius.at<double>(possibleSolutions[i][1], possibleSolutions[i][0]) >= 255) {
                max = houghSpace.at<double>(possibleSolutions[i][1], possibleSolutions[i][0], radius);
                bestRadius = radius;
                bestX = possibleSolutions[i][0];
                bestY = possibleSolutions[i][1];
            }
        }

        if(bestX >= 0 && bestY >= 0)
            deleteNeighbouringCircles(summedRadius, bestY, bestX, 15);

        if (bestRadius >= 0) solution.push_back(Vec3f(bestX, bestY, bestRadius));
    }

    return solution;
}

vector<Vec3f> getHoughCircles(Mat magnitudeImage, Mat directionImage, Mat image, int maximumCircleRadius){


    int houghSpaceSize[] = { image.rows, image.cols, maximumCircleRadius };
    Mat houghSpace(3, houghSpaceSize, CV_64F, Scalar::all(0));
    computeHough3D(image, magnitudeImage, directionImage, houghSpace, maximumCircleRadius);


    Mat summedRadius(2, magnitudeImage.size, CV_64F, Scalar::all(0));
    computeSummedRadius(image, houghSpace, summedRadius, maximumCircleRadius);
    imwrite("2DHough15.jpg", summedRadius);


    vector<Vec3f> possibleSolutions = getPossibleSolutions(image, summedRadius, houghSpace, maximumCircleRadius);

    return selectMostLikelySolutions(possibleSolutions, houghSpace, summedRadius, maximumCircleRadius);

}

bool isNoEntryCorrespondCircle(Rect noEntry, Vec3f circle){
    if((circle[0] > noEntry.x + noEntry.width * 0.3) && (circle[0] <  noEntry.x + noEntry.width * 0.7))
        if((circle[1] > noEntry.y + noEntry.height* 0.3) && (circle[1] <  noEntry.y + noEntry.height * 0.7))
            if(abs(circle[2] - min(noEntry.width, noEntry.height) / 2) < 30) return true;

            return false;
}

void filterNoEntrySigns(vector<Rect> &noEntrySigns, vector<Vec3f> houghCircles){

    vector<Rect> filteredNoEntrySigns;

    for( int i = 0; i < noEntrySigns.size(); i++ )
        for (int j = 0; j < houghCircles.size(); j++)
            if (isNoEntryCorrespondCircle(noEntrySigns[i], houghCircles[j]))
                filteredNoEntrySigns.push_back(Rect(noEntrySigns[i].x, noEntrySigns[i].y, noEntrySigns[i].width, noEntrySigns[i].height));


    noEntrySigns.clear();
    for(int i = 0; i < filteredNoEntrySigns.size(); i++) {
        noEntrySigns.push_back(filteredNoEntrySigns[i]);
    }

}
