#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <cmath>
#include <vector>
#include <fstream>

#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;



void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();

    // esta funcion fue extraida de (http://opencv.willowgarage.com/wiki/cvBlobsLib)

    // lo que realiza es escanear imagen pixel, cuando encuentra elemento que no es fondo y esta sin label, 
    // ocupa floodfill para inundar toda esa zona con conexion por vencidad 4.
    // Finalmente guarda blobs en una estructura.

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

            
            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }
            

            blobs.push_back(blob);

            label_count++;
        }
    }
}


int main( void  ) // funcion principal
{


    VideoCapture cap("Tarea7.wmv"); // abre video   
    double fps = cap.get(CV_CAP_PROP_FPS); // cantidad de fps      
    
    Mat fondo;  // carga de fondo inicial
	fondo = imread( "fondo_estimado.jpg", CV_LOAD_IMAGE_GRAYSCALE);    
    Size size(480,360); // dimencion rescalado imagen
    resize(fondo,fondo,size);// rescalado imagen

    int cont = 0;  //izquierda
    int cont2 = 0;	//derecha


    int flag_k[6] = {0, 0, 0, 0, 0, 0};  // flag para conteo, se cuenta autos por cada carril
    int flag_k1[6] = {0, 0, 0, 0, 0, 0};


    while(1)
    {
        Mat frame;
        cap.read(frame); // lee nuevo frame
        resize(frame,frame,size);

        Mat frame_gris; // pasa a escala de grises
        cvtColor(frame, frame_gris, CV_BGR2GRAY);              
  
	    fondo.convertTo(fondo, CV_32FC1); //-----pasa data a 32 bits, para no tener errores de underflow
        frame_gris.convertTo(frame_gris, CV_32FC1);

        //--------actualizacion modelo del fondo---------
        double alpha = 0.05;
        fondo = (1-alpha)*fondo + alpha*frame_gris;

        //--------estimacion de movimiento---------
        Mat mov = frame_gris-fondo;
        //mov = abs(mov); // opcional, mejora detecccion pero añade efecto de ghosting
        mov = mov > 20;  // binarizacion de imagen de movimiento

        //---operaciones morfologicas para conectar zonas de cada auto y rellenarlas, de forma de  obtener un blob
        mov.convertTo(mov, CV_8UC1);
        Mat morf ; //= mov;
    
 		int erosion_size = 1;  
       	Mat element = getStructuringElement(cv::MORPH_ELLIPSE ,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size) );
 
       	erode(mov,morf,element);  // dilate(image,dst,element);



 		erosion_size = 5;  
       	element = getStructuringElement(cv::MORPH_ELLIPSE ,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size) );

		dilate(morf,morf,element);  // dilate(image,dst,element);

		
		erosion_size = 5;  
       	element = getStructuringElement(cv::MORPH_ELLIPSE ,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size) );

		dilate(morf,morf,element);  // dilate(image,dst,element);

		erosion_size = 5;  
       	element = getStructuringElement(cv::MORPH_ELLIPSE ,
        Size(2 * erosion_size + 1, 2 * erosion_size + 1),
        Point(erosion_size, erosion_size) );

        erode(morf,morf,element);  // dilate(image,dst,element);


        frame_gris.convertTo(frame_gris, CV_8UC1);
        fondo.convertTo(fondo, CV_8UC1);

        //----labeling de blobs
        Mat labels;
        morf = morf/255;
        std::vector < std::vector<cv::Point2i > > blobs;

        FindBlobs(morf ,blobs);

        Mat output = Mat::zeros(morf.size(), CV_8UC3);

        //---tracking de blobs con bounding boxes

         vector<Rect> boun_boxs;
          
          flag_k[0] = 0; 
          flag_k[1] = 0; 
          flag_k[2] = 0;
          flag_k[3] = 0; 
          flag_k[4] = 0;
          flag_k[5] = 0;

         for(size_t i=0; i < blobs.size(); i++) 
         {

                   
           Rect tmp = boundingRect(blobs[i]);
           // estimacion de centros
           Point pt;
		   pt.x = tmp.x + tmp.width/2 ;
		   pt.y = tmp.y + tmp.height/2 ;

           boun_boxs.push_back(tmp);
           if(tmp.y > 155  &&  (tmp.y+tmp.height) < 300  && tmp.width>10 )
           {
           rectangle(frame, tmp , Scalar(0,255,0), 1, 8, 0);
           circle(frame, pt, 1, Scalar(0,255,0), 1, 8, 0);
           }

           int lim_sup = 235;
           int lim_inf = 250;
           int lim_tam= 10;

           
           if(pt.y< lim_inf  &&  pt.y > lim_sup  && pt.x < 470 && pt.x > 30  && tmp.width > lim_tam)  // condiciones zona de deteccion
           {
           rectangle(frame, tmp , Scalar(0,0,255), 3, 6, 0); // 
           circle(frame, pt, 1, Scalar(0,0,255), 2, 8, 0);
           }
           
           // verificador de autos en cada carril

           if(pt.y< lim_inf  &&  pt.y > lim_sup && pt.x < 470   && pt.x > 375 & tmp.width > lim_tam)           
           	flag_k[5] = 1;

           if(pt.y< lim_inf  &&  pt.y > lim_sup && pt.x < 375   && pt.x > 325 & tmp.width > lim_tam)
            flag_k[4] = 1;          

           if(pt.y< lim_inf  &&  pt.y > lim_sup &&  pt.x < 325   && pt.x > 270 & tmp.width > lim_tam)           
             flag_k[3] = 1;    

           if(pt.y< lim_inf  &&  pt.y > lim_sup && pt.x < 220   && pt.x > 160 & tmp.width > lim_tam)           
           	flag_k[2] = 1;

           if(pt.y< lim_inf  &&  pt.y > lim_sup && pt.x < 159   && pt.x > 90 & tmp.width > lim_tam)
            flag_k[1] = 1;          

           if(pt.y< lim_inf  &&  pt.y > lim_sup &&  pt.x < 89   && pt.x > 30 & tmp.width > lim_tam)           
             flag_k[0] = 1;          

          // dibuja de lineas
            line(frame, Point(30, (lim_sup+lim_inf)*0.5) , Point(207,(lim_sup+lim_inf)*0.5), Scalar(0,0,255),2, 8, 0);
           line(frame, Point(270, (lim_sup+lim_inf)*0.5) , Point(470,(lim_sup+lim_inf)*0.5), Scalar(0,0,255),2, 8, 0);


        }
       	

        // chequeado transicion de flag de 0 a 1, se marca como el paso de auto
        if( flag_k1[5] == 0 && flag_k[5] == 1 )
       		cont2 = cont2 +1;             
        
        if( flag_k1[4] == 0 && flag_k[4] == 1 )
       		cont2 = cont2 +1;                  

        if( flag_k1[3] == 0 && flag_k[3] == 1 )
       		cont2 = cont2 +1;

        if( flag_k1[2] == 0 && flag_k[2] == 1 )
       		cont = cont +1;             
        
        if( flag_k1[1] == 0 && flag_k[1] == 1 )
       		cont = cont +1;                  

        if( flag_k1[0] == 0 && flag_k[0] == 1 )
       		cont = cont +1;
                   
       	//alcenado valor anterior de flag
       	flag_k1[5] = flag_k[5];
        flag_k1[4] = flag_k[4];
        flag_k1[3] = flag_k[3];
        flag_k1[2] = flag_k[2];
        flag_k1[1] = flag_k[1];
        flag_k1[0] = flag_k[0];


      	char str[200];
        sprintf(str, "autos izquierda:" "%i",cont);
		putText(frame, str, Point2f(33,25), FONT_HERSHEY_COMPLEX_SMALL, 0.7,  Scalar(0,0,255, 0.5), 1);

      	char str2[200];
        sprintf(str2, "autos derecha:" "%i",cont2);
		putText(frame, str2, Point2f(295,25), FONT_HERSHEY_COMPLEX_SMALL, 0.7,  Scalar(0,0,255, 0.5), 1);


        imshow("deteccion de blobs", frame_gris +morf*255); //
        imshow("tracking +  conteo de blobs", frame ); //
        imshow("detec. objetos en movimiento", mov ); //

        if(waitKey(10) == 27) //se baja el tiempo, para mantener framete de 30 fps considerando tiempo de ejecucion
       {
           cout << "esc key is pressed by user" << endl;
            break; 
       }

    }

    waitKey(0);
    return 0;
}