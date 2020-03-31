
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cxcore.h"
#include <iostream>
#include <math.h>
#include <string.h>
	
#pragma GCC diagnostic ignored "-Wwrite-strings"

using namespace cv;
using namespace std;

static void help()
{
    cout <<
    "\n--------------------------------------------------------&&\n"
    "Function:  License Plate Detection and Localization     &&\n"
    "Aim:       Get ROI of License Plate out                 &&\n"
    "Algorithm: A combination of Gray-scale Mathmetical      &&\n"
    "           Morphological Algorithm and Color-based      &&\n"
    "           License Plate segementation algorithm.       &&\n"
    "--------------------------------------------------------&&\n"
    "Author:    Yohanna WANG                                 &&\n"
    "Date:      2016/5/31                                    &&\n"
    "--------------------------------------------------------&&\n"
    "Using OpenCV version " << CV_VERSION << "\n" << endl;

}

const char* wndname = "Square Detection Demo";

//---------------Contrast Enhancement to Highlight Yellow Range---------------------//
void contrastEnhance( Mat& image, Mat& Enhance, vector<Rect>& squares , vector<Mat>& ceil_img)
{  

       Mat contrast_enhance;
       contrast_enhance.create(image.size(),image.type());
       int nr=image.rows;
       int nc=image.cols*image.channels();
       for(int i=1; i<nr-1; i++)
      {
           const uchar* up_line=image.ptr<uchar>(i-1);  //pointer point to up line
           const uchar* mid_line=image.ptr<uchar>(i);   //pointer point to this line
           const uchar* down_line=image.ptr<uchar>(i+1);//pointer point to next line
           uchar* cur_line=contrast_enhance.ptr<uchar>(i);
           for(int j=1;j<nc-1;j++)
           {
                   cur_line[j]=saturate_cast<uchar>(5*mid_line[j]-
                               mid_line[j-1]-mid_line[j+1]-
                               up_line[j]-down_line[j]);
           //the filter here is[0,-1,0; -1,5,-1; 0,-1,0]; sharp(i,j);
            }
       }
       // set four corners in the "sharp" to be zero;
       contrast_enhance.row(0).setTo(Scalar(0));
       contrast_enhance.row(contrast_enhance.rows-1).setTo(Scalar(0));
       contrast_enhance.col(0).setTo(Scalar(0));
       contrast_enhance.col(contrast_enhance.cols-1).setTo(Scalar(0));
       
       namedWindow("Contrast_enhance",WINDOW_NORMAL);
       imshow("Contrast_enhance",contrast_enhance);
       //imwrite("1.contrast_enhance.jpg",contrast_enhance);
       contrast_enhance.copyTo(Enhance);
}

//---------------------Color Segmentation--------------------------------------//
void colorSegment( Mat& image, Mat& Enhance, vector<Mat>& ceil_img, 
                   int minContour, int maxContour)
{
       ceil_img.clear();
        // Convert input image to HSV
       Mat hsv_image;
       cvtColor(Enhance, hsv_image, COLOR_BGR2HSV);

	// Threshold the HSV image, keep only the red pixels
       Mat yellow_range;
       inRange(hsv_image, cv::Scalar(24, 220, 180), cv::Scalar(32, 255, 255), yellow_range);

       namedWindow("hsv",WINDOW_NORMAL);
       namedWindow("yellow",WINDOW_NORMAL);
       imshow("hsv",hsv_image); 
       imshow("yellow",yellow_range);
       //imwrite("2.hsv.jpg",hsv_image); 
       //imwrite("3.yellow.jpg",yellow_range);

       //Blur the yellow range       
       Mat gauss_diff;
       GaussianBlur(yellow_range,gauss_diff,Size(9,9),3.0);
       namedWindow("gauss_diff",WINDOW_NORMAL);
       imshow("gauss_diff",gauss_diff);
       //imwrite("4.gauss_diff.jpg",gauss_diff);

       //Connect the LP area 
       Mat thresh_diff;
       threshold(gauss_diff,thresh_diff,29,255,THRESH_BINARY);//binary image again
       //namedWindow("thresh_diff",WINDOW_NORMAL);
       //imshow("thresh_diff",thresh_diff);
       //imwrite("5.thresh_diff.jpg",thresh_diff);

       Mat mor_close;
       Mat mor_open;
       Mat element5(6,6,CV_8U,Scalar(1)); //(8,8)
       Mat element = getStructuringElement(MORPH_RECT, Size(32,16) ); //24,4
       morphologyEx(thresh_diff, mor_close, MORPH_CLOSE, element);//close operation
       //namedWindow("closed",WINDOW_NORMAL);
       //imshow("closed",mor_close);
       //imwrite("6.closed.jpg",mor_close);

       morphologyEx(mor_close,mor_open,MORPH_OPEN,element5);//open operation
       //namedWindow("open",WINDOW_NORMAL);
       //imshow("open",mor_open);
       //imwrite("7.open.jpg",mor_open);

       //Filter out pepper and salt noise
       Mat median_diff; 
       medianBlur (mor_open, median_diff, 15);
       //namedWindow("median_diff",WINDOW_NORMAL);
       //imshow("median_diff",median_diff);
       //imwrite("8.median_diff.jpg",median_diff);

        // find contours
       vector<vector<Point> > contours;
       findContours(median_diff, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        /// Approximate contours to polygons + get bounding rects and circles
       vector<vector<Point> > approx( contours.size() );
       vector<Rect> boundRect( contours.size() );

        for( size_t i = 0; i < contours.size(); i++ )         
          { approxPolyDP( Mat(contours[i]), approx[i], 3, true ); 
            
            boundRect[i] = boundingRect( Mat(approx[i]) );

                  int W = boundRect[i].size().width; 
                  int H = boundRect[i].size().height;
                  int bound_area = W * H;
                  float r = W/H;
                  if(r < 1)
                      r = H/ W;

                  float counter_area = fabs(contourArea(Mat(approx[i])));

               if ( fabs(contourArea(Mat(approx[i]))) > minContour && 
                    fabs(contourArea(Mat(approx[i]))) < maxContour &&
                    counter_area / bound_area > 0.3 &&
                    r >= 1 && r <= 7)
                {   
                    //rectangle( image, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );

                     Rect roi= boundRect[i]; //boundingRect( Mat(approx[i]);

                     Mat roi_img = image(roi);

                     ceil_img.push_back(roi_img);
                }
          }

       namedWindow( "Contours", WINDOW_NORMAL);
       imshow("Contours", image);  
       //imwrite("9.Contours.jpg", image);  
}
//------------------------------------------------------------------------------//

void grayProcess( const Mat& image, vector<Mat>& ceil_img, int minContour, int maxContour)
{
        ceil_img.clear();

	Mat img_gray;	
	cvtColor(image, img_gray, COLOR_BGR2GRAY);
	blur(img_gray, img_gray, Size(5,5));
        namedWindow("img_blur",WINDOW_NORMAL);
        imshow("img_blur",img_gray);
        //imwrite("img_blur.jpg",img_gray);    

        // For the vertical edge
	Mat img_sobel;
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
        namedWindow("Sobel",WINDOW_NORMAL);
	imshow("Sobel",img_sobel);
        //imwrite("sobel.jpg",img_sobel);

        // get the threshold and with morphology processing
	Mat img_threshold;
	threshold(img_sobel, img_threshold, 0, 255, THRESH_OTSU+THRESH_BINARY);

        for (int a = 0; a < img_threshold.size().height/6 ; a++ )
            {
               for (int b = 0; b < img_threshold.size().width; b++)
                 {
                  img_threshold.at<uchar>(a,b)= 0;
                 } 
            }

        //namedWindow("Threshold",WINDOW_NORMAL);
        //imshow("Threshold", img_threshold);
        //imwrite("threshold.jpg",img_threshold);

	Mat gauss_diff,threshAgain_diff;
	GaussianBlur(img_threshold,gauss_diff,Size(9,9),3.0);         //Gaussican Blur
        namedWindow("gauss_diff",WINDOW_NORMAL);
	imshow("gauss_diff",gauss_diff);
        //imwrite("guass_diff.jpg",gauss_diff);

	threshold(gauss_diff,threshAgain_diff,29,255,THRESH_BINARY);  //binary image again
        namedWindow("threshAgain_diff",WINDOW_NORMAL);
	imshow("threshAgain_diff",threshAgain_diff);
        //imwrite("threshAgain_diff.jpg",threshAgain_diff);
        
        Mat median_diff; 
        medianBlur ( threshAgain_diff, median_diff, 15);
        namedWindow("median_diff",WINDOW_NORMAL);
        imshow("median_diff",median_diff);
        //imwrite("median_diff.jpg",median_diff);

	Mat mor_close;
	Mat mor_open;
	Mat element5(6,6,CV_8U,Scalar(1)); //(8,8)
	Mat element = getStructuringElement(MORPH_RECT, Size(32,16) ); //24,4
	morphologyEx(median_diff, mor_close, MORPH_CLOSE, element);  //close operation
        namedWindow("closed",WINDOW_NORMAL);
	imshow("closed",mor_close);
        //imwrite("closed.jpg",mor_close);

	morphologyEx(mor_close,mor_open,MORPH_OPEN,element5);         //open operation
        namedWindow("open",WINDOW_NORMAL);
	imshow("open",mor_open);
        //imwrite("open.jpg",mor_open);

        Mat threshold_output;
        threshold( mor_open, threshold_output, 0, 255, THRESH_BINARY);
        //namedWindow("threshold_output",WINDOW_NORMAL);
        //imshow("threshold_output",threshold_output);

        // find contours
        vector<vector<Point> > contours;
        findContours(threshold_output, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        // Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point> > approx( contours.size() );
        vector<Rect> boundRect( contours.size() );
 
        for( size_t i = 0; i < contours.size(); i++ )         
          { approxPolyDP( Mat(contours[i]), approx[i], 3, true ); 
            
            boundRect[i] = boundingRect( Mat(approx[i]) );

                  int W = boundRect[i].size().width; 
                  int H = boundRect[i].size().height;
                         
                  int bound_area = W * H;

                  float counter_area = fabs(contourArea(Mat(approx[i])));
                  float r = W / H;
/*
                  if(r < 1)
                          {
                           r = H / W;
                          }
*/
               if ( fabs(contourArea(Mat(approx[i]))) > minContour   && 
                    fabs(contourArea(Mat(approx[i]))) < maxContour &&                   
                    counter_area / bound_area >= 0.3 )//&&
                  //  r >= 0.7 && r <= 7)
                    //(W/H<6 && W/H >2) || (W/H <1.8 && W/H >0.7))
                {

                 
                     //rectangle( image, boundRect[i].tl(), boundRect[i].br(), Scalar(0,0,255), 2, 8, 0 );

                     Rect roi= boundRect[i]; //boundingRect( Mat(approx[i]);

                     Mat roi_img = image(roi);

                     ceil_img.push_back(roi_img);
                         
                }
          }

       namedWindow( "Contours", WINDOW_NORMAL);
       imshow("Contours", image);  
       //imwrite("Contours.jpg",image);
       
}

//--------------------Extract  ROI with edge information--------------------------//
int extractROI_edge(const Mat& image, vector<Mat>& ceil_img, vector<Mat>& fail_img)
{        
         vector<int>count;
         vector<int>::iterator iter;
         count.push_back(ceil_img.size());
         fail_img.clear();
        
         for(iter = count.begin(); iter!= count.end(); iter++)
         {  
             if(find(count.begin(), count.end(), 0) == count.end() )
             {
             cout<<"The number of ROI in ceil_img = "<<*iter<<endl;
             }
             else
             {
             cout<<"No ROI in ceil_img "<<endl;
             }         
         }

         //The selection of LP in ROI
         for(int k = 0; k < ceil_img.size(); k++ )
         {
             Mat roi_gray;
            
             ceil_img[k].copyTo(roi_gray);

             cvtColor(roi_gray, roi_gray, COLOR_BGR2GRAY);
             //Then find the Sobel edge of ROI
	     Mat roi_sobel;
	     Sobel(roi_gray, roi_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);

             // Get the threshold and with morphology processing
	     Mat roi_threshold;
	     threshold(roi_sobel, roi_threshold, 0, 255, THRESH_OTSU+THRESH_BINARY);

             medianBlur ( roi_threshold, roi_threshold, 3);
             imshow("roi_threshold",roi_threshold);
             //imwrite("10.roi_threshold.jpg",roi_threshold);

                  //int h1 = roi_threshold.size().height/3; 
                      int k1 = 0;
                      int k2 = 0;
                      int k3 = 0;
                      int h1 = roi_threshold.size().height/3;
                      int h2 = roi_threshold.size().height/2; 
                      int h3 = roi_threshold.size().height/3*2; 
                
             for (int w = 0; w < roi_threshold.size().width; w++)
                 {  

                   
                      if (  roi_threshold.at<uchar>(h1,w) == 0 && 
                            roi_threshold.at<uchar>(h1,w+1) == 255  )
                            k1 = k1 + 1;
                      if (  roi_threshold.at<uchar>(h2,w) == 0 && 
                            roi_threshold.at<uchar>(h2,w+1)  == 255  )
                            k2 = k2 + 1;
                      if (  roi_threshold.at<uchar>(h3,w) == 0 && 
                            roi_threshold.at<uchar>(h3,w+1)  == 255  )
                            k3 = k3 + 1; 
                 }
                  cout<<"K1 = "<<k1<<" "<<"K2 = "<<k2<<" "<<"K3 = "<<k3<<endl;
  
                  if (  (k1>1 && k1 <12) && (k2>1 && k2 <15) && (k3>0) )
                 {
                  fail_img.push_back(ceil_img[k]);
                  continue;
                 }
         }
                 
}

//------------------------Extract  ROI with color information---------------------//
int extractROI_color(const Mat& image, vector<Mat>& ceil_img, vector<Mat>& fail_img)
{ 
         vector<int>count;
         vector<int>::iterator iter;
         count.push_back(ceil_img.size());
         fail_img.clear();
                                                                           
         for(iter = count.begin(); iter!= count.end(); iter++)       
         {                                                          
             if(find(count.begin(), count.end(), 0) == count.end() )
             {
             cout<<"The number of ROI in ceil_img = "<<*iter<<endl;
             }
             else
             {
             cout<<"No ROI in ceil_img "<<endl;
             }         
         }
       
         for(int k = 0; k < ceil_img.size(); k++ )
         {
             Mat roi_enhance;
            
             ceil_img[k].copyTo(roi_enhance);       

             int nr=roi_enhance.rows;
             int nc=roi_enhance.cols*roi_enhance.channels();

             for(int i=1; i<nr-1; i++)
                {
                 const uchar* up_line=roi_enhance.ptr<uchar>(i-1);  //pointer point to up line
                 const uchar* mid_line=roi_enhance.ptr<uchar>(i);   //pointer point to this line
                 const uchar* down_line=roi_enhance.ptr<uchar>(i+1);//pointer point to next line
                 uchar* cur_line=roi_enhance.ptr<uchar>(i);
                 for(int j=1;j<nc-1;j++)
                    {
                     cur_line[j]=saturate_cast<uchar>(5*mid_line[j]-
                                 mid_line[j-1]-mid_line[j+1]-
                                 up_line[j]-down_line[j]);
                    }      
                    roi_enhance.row(0).setTo(Scalar(0));
                    roi_enhance.row(roi_enhance.rows-1).setTo(Scalar(0));
                    roi_enhance.col(0).setTo(Scalar(0));
                    roi_enhance.col(roi_enhance.cols-1).setTo(Scalar(0)); 
                } 

         // Convert input image to HSV
         Mat roi_hsv;
         cvtColor(roi_enhance, roi_hsv, COLOR_BGR2HSV);
 
         Mat roi_yellow_range;
         inRange(roi_hsv, cv::Scalar(13, 120, 170), cv::Scalar(41, 255, 255), roi_yellow_range);
         imshow("roi_yellow_range",roi_yellow_range);
         //imwrite("roi_yellow_range.jpg",roi_yellow_range);

         /// This is for the second selection of ROI
         ///Concept is: The ratio of non-zero pixels over zero pixels;

           if(roi_yellow_range.empty())
             { }           
           
             double TotalNumberOfPixels = roi_yellow_range.rows * roi_yellow_range.cols;
             double NonZeroPixels       = countNonZero(roi_yellow_range);
             double ZeroPixels          = TotalNumberOfPixels - NonZeroPixels;
             double Ratio = NonZeroPixels/TotalNumberOfPixels;
             Ratio = ( (double)((int)((Ratio + 0.005)*100) ) )/100;
           
             if(Ratio > 0.12 && Ratio < 0.7)
             {
             //cout<<"non zero Pixels = "<<NonZeroPixels<<"Zero pixels = "<<ZeroPixels<<endl;
             cout<<"NonZero/Total = "<<Ratio<<endl;
             fail_img.push_back(ceil_img[k]); 
             continue;
             }    
      } 

}

//---------------------show Multiple ROI in the same image------------------------//

void showMultiRoI(const Mat& image, vector<Mat>& ceil_img, vector<Mat>& fail_img,
                  int windowHeight, int nRows )
{
          vector<int>c;
          vector<int>::iterator iter;
          c.push_back(ceil_img.size());


          for(iter = c.begin(); iter!= c.end(); iter++)
          {  
              if(find(c.begin(), c.end(), 0) == c.end() )
              
              cout<<"The number of ROI in fail_img = "<<*iter<<endl;
            
              else
              {
              cout<<"No ROI in fail_img "<<endl;

              }         
          }

            int N = ceil_img.size();
            nRows  = nRows > N ? N : nRows; 
            int edgeThickness = 10;
            int imagesPerRow = ceil(double(N) / nRows);
            int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
            int maxRowLength = 0;

            vector<int> resizeWidth;
            for (int i = 0; i < N;)
            {
                    int thisRowLen = 0;
                    for (int k = 0; k < imagesPerRow; k++) 
                    {
                            double aspectRatio = double(ceil_img[i].cols) / ceil_img[i].rows;
                            int temp = int( ceil(resizeHeight * aspectRatio));
                            resizeWidth.push_back(temp);
                            thisRowLen += temp;
                            if (++i == N) break;
                    }
                    if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength)
                    {
                            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
                    }
            }
            int windowWidth = maxRowLength;
            Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

            for (int k = 0, i = 0; i < nRows; i++) 
            {
                    int y = i * resizeHeight + (i + 1) * edgeThickness;
                    int x_end = edgeThickness;
                    for (int j = 0; j < imagesPerRow && k < N; k++, j++) 
                    {
                            int x = x_end;
                            Rect roi(x, y, resizeWidth[k], resizeHeight);
                            Size s = canvasImage(roi).size();

                            // change the number of channels to three
                            Mat target_ROI(s, CV_8UC3);
                            ceil_img[k].copyTo(target_ROI);//---NEWLY added;
                            if (ceil_img[k].channels() != canvasImage.channels())
                            {
                                if (ceil_img[k].channels() == 1) 
                                {
                                    cvtColor(ceil_img[k], target_ROI, COLOR_GRAY2BGR);
                                }
                            }
                            resize(target_ROI, target_ROI, s);
                            if (target_ROI.type() != canvasImage.type())
                            {
                                target_ROI.convertTo(target_ROI, canvasImage.type());
                            }
                            target_ROI.copyTo(canvasImage(roi));
                            x_end += resizeWidth[k] + edgeThickness;
                    }
            }
            if(ceil_img.size() != 0)
            imshow("ceil_Image",canvasImage);
            imwrite("ceil_img.jpg",canvasImage);

}

//---------------------show Single ROI in the same image------------------------//
void showSingleROI(const Mat& image, vector<Mat>& ceil_img, vector<Mat>& fail_img,
                  int windowHeight, int nRows )
{
          vector<int>c;
          vector<int>::iterator iter;
          c.push_back(fail_img.size());


          for(iter = c.begin(); iter!= c.end(); iter++)
          {  
              if(find(c.begin(), c.end(), 0) == c.end() )
              
              cout<<"The number of ROI in fail_img = "<<*iter<<endl;
            
              else
              {
              cout<<"No ROI in fail_img "<<endl;

              }         
          }

            int N = fail_img.size();
            nRows  = nRows > N ? N : nRows; 
            int edgeThickness = 10;
            int imagesPerRow = ceil(double(N) / nRows);
            int resizeHeight = floor(2.0 * ((floor(double(windowHeight - edgeThickness) / nRows)) / 2.0)) - edgeThickness;
            int maxRowLength = 0;

            vector<int> resizeWidth;
            for (int i = 0; i < N;)
            {
                    int thisRowLen = 0;
                    for (int k = 0; k < imagesPerRow; k++) 
                    {
                            double aspectRatio = double(fail_img[i].cols) / fail_img[i].rows;
                            int temp = int( ceil(resizeHeight * aspectRatio));
                            resizeWidth.push_back(temp);
                            thisRowLen += temp;
                            if (++i == N) break;
                    }
                    if ((thisRowLen + edgeThickness * (imagesPerRow + 1)) > maxRowLength)
                    {
                            maxRowLength = thisRowLen + edgeThickness * (imagesPerRow + 1);
                    }
            }
            int windowWidth = maxRowLength;
            Mat canvasImage(windowHeight, windowWidth, CV_8UC3, Scalar(0, 0, 0));

            for (int k = 0, i = 0; i < nRows; i++) 
            {
                    int y = i * resizeHeight + (i + 1) * edgeThickness;
                    int x_end = edgeThickness;
                    for (int j = 0; j < imagesPerRow && k < N; k++, j++) 
                    {
                            int x = x_end;
                            Rect roi(x, y, resizeWidth[k], resizeHeight);
                            Size s = canvasImage(roi).size();

                            // change the number of channels to three
                            Mat target_ROI(s, CV_8UC3);
                            fail_img[k].copyTo(target_ROI);//---NEWLY added;
                            if (fail_img[k].channels() != canvasImage.channels())
                            {
                                if (fail_img[k].channels() == 1) 
                                {
                                    cvtColor(fail_img[k], target_ROI, COLOR_GRAY2BGR);
                                }
                            }
                            resize(target_ROI, target_ROI, s);
                            if (target_ROI.type() != canvasImage.type())
                            {
                                target_ROI.convertTo(target_ROI, canvasImage.type());
                            }
                            target_ROI.copyTo(canvasImage(roi));
                            x_end += resizeWidth[k] + edgeThickness;
                    }
            }
            if(fail_img.size() != 0)
            imshow("detected_Image",canvasImage);
            imwrite("detected_img.jpg",canvasImage);

}

//---------------------save Candidate ROI from the detected vector------------------------//
void SaveMultipleRoI(vector<Mat>& ceil_img, vector<Mat>& fail_img )
{
          for (int roi = 0; roi < fail_img.size() ; roi++)
              {
              stringstream ss;

              string name = "RoI_";
              string type = ".tif";

              ss<<name<<(roi + 1)<<type;

              string filename = ss.str();
              ss.str("");

              imwrite(filename, fail_img[roi]);
              }   
}
//--------------------------------Main Function--------------------------------------------//
int main(int argc , char** argv)
{
/*static*/ const char* names[] = {/* "1.jpg","2.jpg","3.jpg","4.jpg","5.jpg",
                                   "6.jpg","7.jpg","8.jpg","9.jpg","10.jpg",
                                   "11.jpg","12.jpg","13.jpg","14.jpg",*/"15.jpg",
                                   "16.jpg","17.jpg","18.jpg","19.jpg","20.jpg", 
                                   "21.jpg","22.jpg","23.jpg","24.jpg",/*"25.jpg",
                                   "26.jpg","27.jpg","28.jpg","29.jpg","30.jpg",
                                   "31.jpg","32.jpg","33.jpg","34.jpg","35.jpg",
                                   "36.jpg","37.jpg","38.jpg","39.jpg","40.jpg", */
                                    0 };
    help();
    //namedWindow( wndname, WINDOW_NORMAL);
    vector<Rect> squares; 
    vector<Mat> ceil_img;
    vector<Mat> fail_img;
    Mat Enhance;
    int minContour = 800; 
    int maxContour = 24000;
    int windowHeight = 500;
    int nRows = 300;
    int n;
    for( n = 0; names[n] != 0; n++ )
    {
        Mat image = imread(names[n], 1);
        if( image.empty() )
        {
            cout << "Couldn't load " << names[n] << endl;
            continue;
        }

/*//------------------------------------------------------------------------------//--||
                                                                                  //--||
       cout << "What time is it now?" << endl;                                    //--||
       float time;                                                                //--||
       cin >> time;                                                               //--||
       int y;                                                                     //--||
          if( time >= 9.00 && time <= 17.00 )                                     //--||
              y = 0;                                                              //--||
          else if ( (time >= 17 && time <= 23 )                                   //--||
                   || (time >= 0  && time <=8))                                   //--||
              y = 1;                                                              //--||
                                                                                  //--||
       switch (y)                                                                 //--||
       {                                                                          //--||   
            case 0:                                                               //--||
                                                                                  //--||
            {                                                                     //--||
            cout <<"We need to use Color-based VLP system"<< endl;                //--||               
            contrastEnhance  (image, Enhance, squares, ceil_img);                 //--||
            colorSegment     (image, Enhance, ceil_img, minContour, maxContour);  //--||
            extractROI_edge  (image, ceil_img, fail_img);                         //--||
            showMultiROI     (image, ceil_img, fail_img, windowHeight, nRows);    //--||
            break;                                                                //--||
            }                                                                     //--||
                                                                                  //--||
            case 1:                                                               //--||
            {                                                                     //--||
            cout <<"We need to use Gray-scale based VLP system"<< endl;           //--||
            grayProcess      (image, ceil_img, minContour, maxContour);           //--||
            extractROI_color (image, ceil_img, fail_img);                         //--||
            showMultiROI     (image, ceil_img, fail_img, windowHeight, nRows);    //--||
            }                                                                     //--||
                                                                                  //--||                                                          
     }                                                                            //--||
*///------------------------------------------------------------------------------//--||
/*
            contrastEnhance  (image, Enhance, squares, ceil_img);                 //--||
            colorSegment     (image, Enhance, ceil_img, minContour, maxContour);  //--||
            if(ceil_img.size()<3 && ceil_img.size()>0)
            {
            showMultiRoI     (image, ceil_img, fail_img, windowHeight, nRows);    //--||
            cout<<"This is the situation of roi<3"<<endl;
            }
            else if(ceil_img.size()>=3)
            {
            extractROI_edge  (image, ceil_img, fail_img);                         //--||
            showSingleROI    (image, ceil_img, fail_img, windowHeight, nRows);    //--||
            showMultiRoI     (image, ceil_img, fail_img, windowHeight, nRows);    //--||
            cout<<"This is the situation of roi>=3"<<endl;
            }
            else if(ceil_img.size()<1)
            {
            cout<<"No ceil_img in this situation"<<endl;
            }            

 */
            grayProcess      (image, ceil_img, minContour, maxContour);           //--||
            extractROI_color (image, ceil_img, fail_img);                         //--||
            showMultiRoI     (image, ceil_img, fail_img, windowHeight, nRows);    //--||
            showSingleROI    (image, ceil_img, fail_img, windowHeight, nRows);    //--||
            SaveMultipleRoI  (ceil_img, fail_img);
        
        int c = waitKey();
        if( (char)c == 27 )
            break;
    }

    return 0;
}
