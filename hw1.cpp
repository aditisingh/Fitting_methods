#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    if( argc != 3)
    {
     cout <<" Usage: hw1 I1 I2" << endl;
     return -1;
    }

    Mat I1, I2;

    I1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);   // Read first file
    I2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);   //Read second file
    
    namedWindow( "Input Image 1", WINDOW_NORMAL);
    imshow( "Input Image 1", I1 ); 

    namedWindow("Input Image 2", WINDOW_NORMAL); 
    imshow("Input Image 2", I2);

    //store all pixel intensities for image 1 and image 2
    int array1[I1.rows*I1.cols];
    int array2[I2.rows*I2.cols];

    //both images are equal-sized
    for( int y=0; y<I1.rows;y++)
    {
        for(int x=0;x<I1.cols;x++)
        {
            array1[x+y*I1.cols]=I1.at<uchar>(y,x);
            array2[x+y*I1.cols]=I2.at<uchar>(y,x);
        }
    }

    //Plotting the data points
    Mat plot(260,260, CV_8UC3, Scalar(255,255,255));

    for(int i=0;i<I1.rows*I1.cols;i++) 
    {
        plot.at<Vec3b>(array1[i],array2[i])[0]=255;
        plot.at<Vec3b>(array1[i],array2[i])[1]=0;
        plot.at<Vec3b>(array1[i],array2[i])[2]=0;
    }

    namedWindow("Plot",WINDOW_NORMAL);
    imshow("Plot",plot);  

    //Fitting line as per total least squares
    Mat plot_fit1=plot.clone(); 
    //Get all datapoints in the vector
    std::vector<Point> datapoints;
    for(int i=0;i<I1.rows*I1.cols;i++)
    {
        datapoints.push_back(Point(array2[i],array1[i])); 
    }

    Vec4f line1;
    fitLine(datapoints,line1,CV_DIST_L2,0,0.01,0.01);
    //plotting line 1
    //find the extreme points
    //equation of line: y-line1[3]=(line1[1]/line1[0])(x-line1[2])
    int y_1=line1[3]-line1[2]*((line1[1]/line1[0]));
    int y_2=line1[3]+(plot_fit1.cols-line1[2])*((line1[1]/line1[0]));

    cout<<"Total least squares line: "<<line1[0]<<" "<<line1[1]<<" "<<line1[2]<<" "<<line1[3]<<endl;

    line(plot_fit1,Point(0,y_1),Point(plot_fit1.cols-1,y_2),Scalar(0,0,255),1,8,0);

    //Robust estimators: Trying  FAIR, WELSCH, HUBER

    Vec4f line2;
    fitLine(datapoints,line2,CV_DIST_HUBER,0,0.01,0.01);
    //plotting line
    //find the extreme points
    //equation of line: y-line[3]=(line[1]/line[0])(x-line[2])
    int y_3=line2[3]-line2[2]*((line2[1]/line2[0]));
    int y_4=line2[3]+(plot_fit1.cols-line2[2])*((line2[1]/line2[0]));

    cout<<"Robust estimators line: "<<line2[0]<<" "<<line2[1]<<" "<<line2[2]<<" "<<line2[3]<<endl;

    line(plot_fit1,Point(0,y_3),Point(plot_fit1.cols-1,y_4),Scalar(0,255,0),1,8,0);

    //Fitting datapoints to Gaussian Model
    double pt_mean_x=0,pt_mean_y=0;
    double cov_XY=0;
    double cov_XX=0;
    double cov_YY=0;

    for(int i=0;i<datapoints.size();i++)
    {
       pt_mean_x=pt_mean_x+datapoints[i].x;
       pt_mean_y=pt_mean_y+datapoints[i].y;
    }

    pt_mean_x=pt_mean_x/(datapoints.size());
    pt_mean_y=pt_mean_y/(datapoints.size());

    for(int i=0;i<datapoints.size();i++)
    {
        cov_XY=cov_XY+(datapoints[i].x-pt_mean_x)*(datapoints[i].y-pt_mean_y);
        cov_XX=cov_XX+(datapoints[i].x-pt_mean_x)*(datapoints[i].x-pt_mean_x);
        cov_YY=cov_YY+(datapoints[i].y-pt_mean_y)*(datapoints[i].y-pt_mean_y);

    }

    cov_XX=cov_XX/(datapoints.size()-1);
    cov_XY=cov_XY/(datapoints.size()-1);
    cov_YY=cov_YY/(datapoints.size()-1);

    double rho=cov_XY/sqrt(cov_XX*cov_YY);

    cout<<"Parameters of gaussian: ";
    cout<<"mean: ("<<pt_mean_x<<","<<pt_mean_y<<")"<<" ,Cov XX: "<<cov_XX<<" ,Cov XY: "<<cov_XY<<" ,Cov YY: "<<cov_YY<<" ,rho: "<<rho<<endl;

    double start_angle=0;
    double end_angle=360;

    // cout<<"Size: "<<sqrt(cov_XX)<<" x "<<sqrt(cov_YY)<<endl;
    
    Mat covmat = (Mat_<double>(2,2) << cov_XX, cov_XY, cov_XY, cov_YY);
    Mat eigenvalues, eigenvectors;
    eigen(covmat, eigenvalues, eigenvectors);

    //Calculate the angle between the largest eigenvector and the x-axis
    double angle = atan2(eigenvectors.at<double>(0,1), eigenvectors.at<double>(0,0));

    //Shift the angle to the [0, 2*pi] interval instead of [-pi, pi]
    if(angle < 0)
        angle += 2*M_PI;

    //Conver to degrees instead of radians
    angle = 180*angle/M_PI;
    cout<<"angle: "<<angle<<endl;


    namedWindow("Plot: Line fit",WINDOW_NORMAL);
    imshow("Plot: Line fit",plot_fit1);


    //Thresholding

    Mat plot_thresh1=plot.clone();
    Mat plot_thresh2=plot.clone();

    double thresh1, thresh2;
    vector<double> dist1;
    std::vector<double> dist2;

    double mean_dist1=0, mean_dist2=0;

    vector<int> indices_lines1;
    vector<int> indices_lines2;

    //finding mean distance for both lines
    for(int i=0;i<datapoints.size();i++)
    {
        dist1.push_back((abs(line1[1]*datapoints[i].x-line1[0]*datapoints[i].y+line1[0]*line1[3]-line1[1]*line1[2])/sqrt(line1[1]*line1[1]+line1[0]*line1[0])));
        dist2.push_back((abs(line2[1]*datapoints[i].x-line2[0]*datapoints[i].y+line2[0]*line2[3]-line2[1]*line2[2])/sqrt(line2[1]*line2[1]+line2[0]*line2[0])));

        mean_dist1+=dist1[i];
        mean_dist2+=dist2[i];
    }
    mean_dist1=mean_dist1/datapoints.size();
    mean_dist2=mean_dist2/datapoints.size();

    // cout<<mean_dist<<endl;
    //Setting threshold as twice of mean distance

    thresh1=2*mean_dist1;
    thresh2=2*mean_dist2;

    //Plot the outliers, and save their indices
    for(int i=0;i<datapoints.size();i++)
    {
        if(dist1[i]<=thresh1)
            plot_thresh1.at<Vec3b>(datapoints[i].y,datapoints[i].x)=Vec3b(255,255,255);
        else
            indices_lines1.push_back(i);

        if(dist2[i]<=thresh2)
            plot_thresh2.at<Vec3b>(datapoints[i].y,datapoints[i].x)=Vec3b(255,255,255);
        else
            indices_lines2.push_back(i);
    }

    namedWindow("Plot: line threshold least sq",WINDOW_NORMAL);
    imshow("Plot: line threshold least sq",plot_thresh1);

    namedWindow("Plot: line threshold robust",WINDOW_NORMAL);
    imshow("Plot: line threshold robust",plot_thresh2);

    //gaussian thresholding 

    double chi_param=5.991;//7.738 for 97.5% confidence, 4.605 for 90% confidence, 5.991 for 95% confidence, 9.210 for 99% confidence
    Size sz=Size(sqrt(eigenvalues.at<double>(0)*chi_param),sqrt(eigenvalues.at<double>(1)*chi_param));
    ellipse(plot_fit1,Point((int)pt_mean_x,(int)pt_mean_y),sz,angle,start_angle,end_angle,Scalar(0,0,0),2,8,0);

    namedWindow("Plot: ellipse",WINDOW_NORMAL);
    imshow("Plot: ellipse",plot_fit1);

    vector<int> indices_gauss;

    Mat plot_gauss_thresh=plot.clone();

    double val;
    angle=angle*M_PI/180;

    //outliers plotted and saved indices
    for(int i=0; i<datapoints.size();i++)
    {
        val=pow((cos(angle)*(datapoints[i].x-pt_mean_x)+sin(angle)*(datapoints[i].y-pt_mean_y)),2)/(eigenvalues.at<double>(0)*chi_param) + pow((sin(angle)*(datapoints[i].x-pt_mean_x)-cos(angle)*(datapoints[i].y-pt_mean_y)),2)/(eigenvalues.at<double>(1)*chi_param);
        if(val<=1)
            plot_gauss_thresh.at<Vec3b>(datapoints[i].y,datapoints[i].x)=Vec3b(255,255,255);
        else
            indices_gauss.push_back(i);
    }
    // ellipse(plot_gauss_thresh,Point((int)pt_mean_x,(int)pt_mean_y),sz,angle*180/M_PI,start_angle,end_angle,Scalar(0,0,0),2,8,0);

    namedWindow("Plot: gaussian dist threshold",WINDOW_NORMAL);
    imshow("Plot: gaussian dist threshold",plot_gauss_thresh);

    // cout<<thresh;

    Mat I3(I1.rows, I1.cols, CV_8UC1, Scalar(0));
    Mat I4(I2.rows, I2.cols, CV_8UC1, Scalar(0)); 
    Mat I5(I1.rows, I1.cols, CV_8UC1, Scalar(0));
    //after noise removal

    int x_coord, y_coord;
    int cols=I1.cols; 
    int rows=I1.rows;
    int index;

    //Plotting the pixels that changed in the three cases
    for(int j=0;j<indices_lines1.size();j++)
    {
        index=indices_lines1[j];
        x_coord=index%cols;
        y_coord=index/cols;

        I3.at<uchar>(y_coord,x_coord)=255;//datapoints[index].y;
    }

    for(int j=0;j<indices_lines2.size();j++)
    {
        index=indices_lines2[j];
        x_coord=index%cols;
        y_coord=index/cols;

        I4.at<uchar>(y_coord,x_coord)=255;//datapoints[index].y;
    }

    for(int j=0;j<indices_gauss.size();j++)
    {
        index=indices_gauss[j];
        x_coord=index%cols;
        y_coord=index/cols;

        I5.at<uchar>(y_coord,x_coord)=255;//datapoints[index].y;
    }

    //Noise removal
    Mat I6=I3.clone();
    Mat I7=I4.clone();
    Mat I8=I5.clone();

    //structuring element

    int erosion_size1=1;
    Mat element1=getStructuringElement(MORPH_ELLIPSE,Size(2*erosion_size1+1,2*erosion_size1+1),Point(erosion_size1,erosion_size1));

    //erode the image
    erode(I3,I6,element1);
    erode(I4,I7,element1);
    erode(I5,I8,element1);

    //remove the salt pepper noise, for line fit results
    medianBlur(I7,I7,3);
    medianBlur(I6,I6,3);

    namedWindow("Plot: Least squares, direct", WINDOW_NORMAL);
    imshow("Plot: Least squares, direct",I3);

    namedWindow("Plot: robust estimators, direct",WINDOW_NORMAL);
    imshow("Plot: robust estimators, direct",I4);


    namedWindow("Plot: Gaussian,direct", WINDOW_NORMAL);
    imshow("Plot: Gaussian,direct",I5);

    namedWindow("Plot: Least squares, filtered",WINDOW_NORMAL);
    imshow("Plot: Least squares, filtered",I6);

    namedWindow("Plot: robust estimators, filtered",WINDOW_NORMAL);
    imshow("Plot: robust estimators, filtered",I7);

    namedWindow("Plot: Gaussian,filtered", WINDOW_NORMAL);
    imshow("Plot: Gaussian,filtered",I8);

    waitKey(0);// Wait for a keystroke in the window
    return 0;
}