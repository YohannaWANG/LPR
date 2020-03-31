#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

static void help()
{
}

int thresh = 50, N = 11;
const char* wndname = "Square Detection Demo";
//----------------Here is the program oF part 1---------------------------------//
//--------------------Image pre-processing--------------------------------------//
//----------------RGB-GRAY--&&--Gamma Correction--------------------------------//

void GammaCorrection(Mat& img, Mat& GammaCorrect, float fGamma)
{
       Mat sharp;
       sharp.create(img.size(),img.type());
       int nr=img.rows;
       int nc=img.cols*img.channels();
       for(int i=1; i<nr-1; i++)
      {
           const uchar* up_line=img.ptr<uchar>(i-1);//pointer point to up line
           const uchar* mid_line=img.ptr<uchar>(i);//pointer point to this line
           const uchar* down_line=img.ptr<uchar>(i+1);//pointer point to next line
           uchar* cur_line=sharp.ptr<uchar>(i);
           for(int j=1;j<nc-1;j++)
           {
                   cur_line[j]=saturate_cast<uchar>(5*mid_line[j]-
                               mid_line[j-1]-mid_line[j+1]-
                               up_line[j]-down_line[j]);
           //the filter here is[0,-1,0; -1,5,-1; 0,-1,0]; sharp(i,j);
            }
       }
       // set four corners in the "sharp" to be zero;
       sharp.row(0).setTo(Scalar(0));
       sharp.row(sharp.rows-1).setTo(Scalar(0));
       sharp.col(0).setTo(Scalar(0));
       sharp.col(sharp.cols-1).setTo(Scalar(0));
       imshow("sharp",sharp);
       imwrite("1.sharp.jpg",sharp);
//-------------------------------------------------------------------------------
       Mat src;  
       cvtColor(sharp, src, COLOR_BGR2GRAY);

	   CV_Assert(src.data);
	   // accept only char type matrices
	   CV_Assert(src.depth() != sizeof(uchar));

	   // build look up table
	   unsigned char lut[256];
	   for (int i = 0; i < 256; i++)
	    {
		  lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	    }

	   GammaCorrect = src.clone();
       MatIterator_<uchar> it, end;
	   for (it = GammaCorrect.begin<uchar>(), end = GammaCorrect.end<uchar>(); it != end; it++)
		   //*it = pow((float)(((*it))/255.0), fGamma) * 255.0;
		    *it = lut[(*it)];

	   imshow("1", img);
	   imshow("GammaCorrect", GammaCorrect);
       imwrite("2.gamma.jpg", GammaCorrect);
}
//---------------------------Gaussian BLUR and Sharping------------------------//
void PreProcessing(Mat& img, Mat& GammaCorrect, Mat& threshold_result)
{

       Mat gauss;
       GaussianBlur(GammaCorrect, gauss, Size(3,3),3.0);//Gaussian Blur with binary image //(Size(9,9))
       imshow("gauss",gauss);
       imwrite("3.Gaussian.jpg",gauss);

       ///Sharpening
       Mat sharp;
       sharp.create(gauss.size(),gauss.type());
       int nr=gauss.rows;
       int nc=gauss.cols*gauss.channels();
       for(int i=1; i<nr-1; i++)
      {
           const uchar* up_line=gauss.ptr<uchar>(i-1);//pointer point to up line
           const uchar* mid_line=gauss.ptr<uchar>(i);//pointer point to this line
           const uchar* down_line=gauss.ptr<uchar>(i+1);//pointer point to next line
           uchar* cur_line=sharp.ptr<uchar>(i);
           for(int j=1;j<nc-1;j++)
           {
                   cur_line[j]=saturate_cast<uchar>(5*mid_line[j]-
                               mid_line[j-1]-mid_line[j+1]-
                               up_line[j]-down_line[j]);
           //the filter here is[0,-1,0; -1,5,-1; 0,-1,0]; sharp(i,j);
            }
       }
       // set four corners in the "sharp" to be zero;
       sharp.row(0).setTo(Scalar(0));
       sharp.row(sharp.rows-1).setTo(Scalar(0));
       sharp.col(0).setTo(Scalar(0));
       sharp.col(sharp.cols-1).setTo(Scalar(0));
       imshow("sharp2",sharp);
       imwrite("4.sharp2.jpg",sharp);

//--------------------Show Histogram of above---------------------------------//
       ///Histogram equalization;
       vector<Mat> g_planes;
       split( sharp, g_planes );
       int histSize = 256;
       float range[] = { 0, 256 } ;
       const float* histRange = { range };

       bool uniform = true; bool accumulate = false;

       Mat g_hist;

       /// Compute the histograms:
       calcHist( &g_planes[0], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );

       // Draw the histograms for B, G and R
       int hist_w = 512; int hist_h = 400;
       int bin_w = cvRound( (double) hist_w/histSize );

       Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

       /// Normalize the result to [ 0, histImage.rows ]
       normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

       /// Draw for each channel
       for( int i = 1; i < histSize; i++ )
       {

       line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                        Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                        Scalar( 0, 255, 0), 2, 8, 0  );

       }

       /// Display
       namedWindow("calcHist", WINDOW_AUTOSIZE );
       imshow("calcHist", histImage );
//---------------Global edge detection--------------------------------------//
/*
       ///Canny edge detection
       Mat detected_edges;
       int lowThreshold=100;
       int ratio = 3;
       int kernel_size = 3;
       Canny( sharp, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
       namedWindow("detected_edges", WINDOW_AUTOSIZE );
       imshow("detected_edges", detected_edges );
*/
       Mat LP_threshold;
       threshold(sharp, LP_threshold, 0, 255, THRESH_OTSU+THRESH_BINARY);
       //imwrite("LP_threshold_9.tif",LP_threshold);
       //imwrite("LP_threshold.png",LP_threshold);
/*
       Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));   
       morphologyEx(LP_threshold,LP_threshold, MORPH_CLOSE, element); 
 
       //medianBlur(LP_threshold,LP_threshold,7);


        for (int a = 0;a < LP_threshold.size().height/30 ; a++ )
            {
               for (int b = 0; b < LP_threshold.size().width; b++)
                 {
                  LP_threshold.at<uchar>(a,b)= 255;
                 } 
            }

        for (int c = 0;c < LP_threshold.size().height; c++ )
            {
               for (int d = 0; d < LP_threshold.size().width; d++)
                 {
               if   (d < LP_threshold.size().width/1.1)
                    {}               
               if   (d >= LP_threshold.size().width/1.1 && d < LP_threshold.size().width)
               LP_threshold.at<uchar>(c,d)= 255;
                 }
            }

        for (int e = 0;e < LP_threshold.size().height; e++ )
            {
               for (int f = 0; f < LP_threshold.size().width/25; f++)
                 {
                  LP_threshold.at<uchar>(e,f)= 255;
                 } 
            }
 
*/     
       threshold_result.push_back(LP_threshold);
       imshow("LP_threshold",LP_threshold);
       imwrite("5.Lp_threshold.jpg",LP_threshold);
}
//----------------Here is the program of part 2----------------------------//
//-----------------Edge remove and perspective correction------------------//
//-------------------------------------------------------------------------//
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
void findSquares( Mat& threshold_result, vector<vector<Point> >& squares )
{
    squares.clear();

    Mat pyr, timg, gray0(threshold_result.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(threshold_result, pyr, Size(threshold_result.cols/2, threshold_result.rows/2));
    pyrUp(pyr, timg, threshold_result.size());
    vector<vector<Point> > contours;
    vector< Vec4i > hierarchy;
    Mat dst(threshold_result.rows,threshold_result.cols,CV_8UC1,Scalar::all(0)); //create destination image
    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        if(gray0.type() == CV_8UC1)

        { cout<<"This is single channel"<<endl;}

        else if(gray0.type() == CV_8UC3)  
        {
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        }
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            if( l == 0 )
            {

                Canny(gray0, gray, 0, thresh, 5);
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
            {
                gray = gray0 >= (l+1)*255/N;
            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            vector<vector<Point> > contours_poly(1);
            //vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                approxPolyDP(Mat(contours[i]), contours_poly[0], arcLength(Mat(contours[i]), true)*0.02, true);

                if(// contours_poly[0].size() == 4 &&
                    fabs(contourArea(Mat( contours_poly[0]))) > 400 &&
                    isContourConvex(Mat( contours_poly[0])) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        double cosine = fabs(angle( contours_poly[0][j%4], contours_poly[0][j-2], contours_poly[0][j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }
                    if( maxCosine < 0.3 )
                        drawContours( dst,contours, i, Scalar(255,255,255),FILLED, 8, hierarchy );
                        
//-----------------------------------------------------------------------------------------------------
                        Rect boundRect = boundingRect(contours[i]);

                        vector<Point2f> quad_pts;
                        vector<Point2f> square_pts;

                        quad_pts.push_back(Point2f(contours_poly[0][0].x,contours_poly[0][0].y));
                        quad_pts.push_back(Point2f(contours_poly[0][1].x,contours_poly[0][1].y));
                        quad_pts.push_back(Point2f(contours_poly[0][3].x,contours_poly[0][3].y));
                        quad_pts.push_back(Point2f(contours_poly[0][2].x,contours_poly[0][2].y));
                        square_pts.push_back(Point2f(boundRect.x,boundRect.y));
                        square_pts.push_back(Point2f(boundRect.x,boundRect.y+boundRect.height));
                        square_pts.push_back(Point2f(boundRect.x+boundRect.width,boundRect.y));
                        square_pts.push_back(Point2f(boundRect.x+boundRect.width,boundRect.y+boundRect.height));

                        Mat transmtx = getPerspectiveTransform(quad_pts,square_pts);
                        Mat transformed = Mat::zeros(threshold_result.rows, threshold_result.cols, CV_8UC3);
                        warpPerspective(threshold_result, transformed, transmtx, threshold_result.size());

//-----------------------------------------------------------------------------------------------------
/*
    Point P1=contours_poly[0][0];
    Point P2=contours_poly[0][1];
    Point P3=contours_poly[0][2];
    Point P4=contours_poly[0][3];


    line(threshold_result,P1,P2, Scalar(0,0,255),1,LINE_AA,0);
    line(threshold_result,P2,P3, Scalar(0,0,255),1,LINE_AA,0);
    line(threshold_result,P3,P4, Scalar(0,0,255),1,LINE_AA,0);
    line(threshold_result,P4,P1, Scalar(0,0,255),1,LINE_AA,0);
    //rectangle(image,boundRect,Scalar(0,255,0),1,8,0);
    rectangle(transformed,boundRect,Scalar(0,255,0),1,8,0);
    //namedWindow("quadrilateral",WINDOW_NORMAL);
    //namedWindow("dst",WINDOW_NORMAL);
    //namedWindow("src",WINDOW_NORMAL);
*/
//------------------------------------------------------------------------
               Mat roi_img;              
               transformed(boundRect).copyTo(roi_img);              
               namedWindow("roi_image",WINDOW_NORMAL);
               imshow("roi_image",roi_img);              
               imwrite("7.roi_image.jpg" ,roi_img);

//------------------------------------------------------------------------

    imshow("quadrilateral", transformed);
    imwrite("6.transformed.jpg",transformed);
    //imshow("dst",dst);
    //imshow("src",image);
//------------------------------------------------------------------------
                        squares.push_back( contours_poly[0]);
                }
            }
        }
    }
}

// the function draws all the squares in the image
void drawSquares( Mat& threshold_result,const vector<vector<Point> >& squares ) 
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];
        int n = (int)squares[i].size();
        polylines(threshold_result, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    imshow(wndname, threshold_result);
}
//-------------------------------------------------------------------------//
int main(int argc, const char ** argv)
{
    static const char* names[] = { "1.jpg",/*"3.jpg","4.jpg","5.jpg",
                                   "6.jpg","7.jpg","8.jpg","9.jpg",*/
                                    0 };

    namedWindow( wndname, WINDOW_NORMAL);
    vector<vector<Point> > squares;

    for( int n = 0; names[n] != 0; n++ )
    {
        Mat img = imread(names[n], 1);
        if( img.empty() )
        {
            cout << "Couldn't load " << names[n] << endl;
            continue;
        }
	//resize(img, img, Size(0, 0), 0.5, 0.5);

	    Mat GammaCorrect(img.rows, img.cols, img.type());
        Mat threshold_result;
	    GammaCorrection(img, GammaCorrect, 2);  
        PreProcessing  (img, GammaCorrect,threshold_result);
        findSquares(threshold_result, squares);
        drawSquares(threshold_result, squares);

        waitKey(0);
    }

  return 0;
}
