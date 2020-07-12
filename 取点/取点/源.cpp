#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>

using namespace std;
using namespace cv;


IplImage* src = 0;

void on_mouse(int event, int x, int y, int flags, void* ustc)

{

	CvFont font;

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);



	if (event == CV_EVENT_LBUTTONDOWN)

	{

		CvPoint pt = cvPoint(x, y);

		char temp[16];

		sprintf(temp, "(%d,%d)", pt.x, pt.y);

		//cvPutText(src, temp, pt, &font, cvScalar(255, 255, 255, 0));

		cvCircle(src, pt, 1, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);

		cvShowImage("src", src);

	}

}



int main()

{

	src = cvLoadImage("Contours.jpg");



	namedWindow("src", WINDOW_NORMAL);
	cvResizeWindow("src", 800, 800); //创建一个500*500大小的窗口
	cvSetMouseCallback("src", on_mouse, 0);


	cvShowImage("src", src);
	cvWaitKey(0);

	cvDestroyAllWindows();

	cvReleaseImage(&src);



	return 0;

}