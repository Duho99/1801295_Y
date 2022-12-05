#include <iostream>     //<iostream>����
#include <opencv2/opencv.hpp>     //<opencv2/opencv.hpp>����
#include <stdio.h>     //<stdio.h>����
#include <Windows.h>     //<Windows.h>����
#include <conio.h>     //<conio.h>����
#include <mmsystem.h>     //<mmsystem.h>����
#pragma comment(lib,"winmm.lib")  //winmm.lib ���̺귯�� �߰�
using namespace std;  //std ����
using namespace cv;  //cv ����
using namespace cv::dnn;  //dnn ��� ����
const int NUM_CLASSES = 7;  //Ŭ���� ����
const Scalar colors[] = {{0, 255, 255},{255, 255, 0},{0, 255, 0},{255, 0, 0}};  //��ü �迭 ����
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);  //�÷� ��ȣ
void sound(string name);  // ���� �޼��� ����Լ� ����
int main() {  //�����Լ�
	vector <string> class_names = { "thirty", "fifty", "sixty", "speedbump", "leftban", "U_turn", "car" };  //Ŭ���� �̸� ����
	auto net = cv::dnn::readNetFromDarknet("yolov4-traffic.cfg", "yolov4-traffic_final.weights");  //�̸� �н��� �𵨷� Net��ü ����
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // ��꿡 opencv�� ����ϵ�����
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);  // ��꿡 CPU�� ����ϵ�����
	auto output_names = net.getUnconnectedOutLayersNames(); //������� ���� ���̾��� �̸��� ��ȯ
	for (int i = 1; i <= 7; i++) {  //�ݺ���
		string filename = format("test%d.jpg", i);   // �����̸� ����
		Mat frame = imread(filename);  // �׽�Ʈ ������ �б�
		Mat blob;  // blob��ü ����
		vector<Mat> detections;  //���ؼ� ��ü ����
		if (frame.empty()){		// �׽�Ʈ ������ ���� ���ǹ�	
			cerr << "frame empty" << endl; return -1;}  //���� �޼��� ���
		cv::dnn::blobFromImage(frame, blob, 1 / 255.f, Size(320,320), Scalar(), true, false, CV_32F);  //�Է¿������� ���� 4���� ��� ��ü ��ȯ
		net.setInput(blob);  //��� ��ü�� ��Ʈ��ũ �Է����� ����
		TickMeter tm;   // ��ü ����
		tm.start();  //�ð� ���� ����
		net.forward(detections, output_names);  //�߷��Ͽ� ����� ��ü�� ��ȯ
		tm.stop();  //�ð� ���� ����
		cout << "time : " << tm.getTimeMilli() << endl;  //�߷� �ð� ���
		vector<int> indices[NUM_CLASSES];  // Ŭ���� �ε���
		vector<cv::Rect> boxes[NUM_CLASSES];  //Ŭ���� �ٿ�� �ڽ�
		vector<float> scores[NUM_CLASSES];   //Ŭ���� �ŷڵ�
		for (auto& output : detections)  //�ݺ���
		{
			const auto num_boxes = output.rows; // �߷е� ��ü ��
			for (int i = 0; i < num_boxes; i++)  // ���� �ݺ���
			{
				auto x = output.at<float>(i, 0) * frame.cols;   //�ٿ�� �ڽ� �߽���ǥ
				auto y = output.at<float>(i, 1) * frame.rows;   //�ٿ�� �ڽ� �߽���ǥ
				auto width = output.at<float>(i, 2) * frame.cols;   //�ٿ�� �ڽ� ��
				auto height = output.at<float>(i, 3) * frame.rows;   //�ٿ�� �ڽ� ����
				Rect rect(x - width / 2, y - height / 2, width, height);  //�ٿ�� �ڽ� Rect ��ü
				for (int c = 0; c < NUM_CLASSES; c++)  //class X �ݺ���
				{
					auto confidence = *output.ptr<float>(i, 5 + c);   //�ٿ�� �ڽ��ȿ� class X ��ü�� ������ Ȯ��
					if (confidence >= 0.5)  // Ȯ�� �ŷڵ��� 0.5���� ũ��
					{
						boxes[c].push_back(rect); //�ٿ�� �ڽ� ���� ����
						scores[c].push_back(confidence);  //�ŷڵ� ����
					}
				}
			}
		}
		for (int c = 0; c < NUM_CLASSES; c++)    //�ݺ���
			cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, 0.5, indices[c]);   //�־��� ���ڿ� �ش� ������ �ƴ� �ִ� ������ ����
		for (int c = 0; c < NUM_CLASSES; c++)    //�ݺ���
		{
			for (int i = 0; i < indices[c].size(); ++i)  //�ݺ���
			{
				const auto color = colors[c % NUM_COLORS];  //�÷�����
				auto idx = indices[c][i];  //�ε���
				const auto& rect = boxes[c][idx];  //�ٿ�� �ڽ�
				rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), color, 3);  //�簢�� �׸��� (�ٿ�� �ڽ�)
				string label_str = class_names[c] + ": " + format("%.02lf", scores[c][idx]);  //Ŭ������� �ŷڵ� ���ڿ�
				int baseline; //��������
				auto label_bg_sz = getTextSize(label_str, FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);  //label_str �ؽ�Ʈ ������
				rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), 
					Point(rect.x + label_bg_sz.width, rect.y), color, FILLED);  //�簢�� �׸��� (Ŭ���� �̸�, �ŷڵ�)
				putText(frame, label_str, Point(rect.x, rect.y - baseline - 5),
					FONT_HERSHEY_COMPLEX_SMALL, 1,Scalar(0, 0, 0));  //Ŭ������� �ŷڵ� ���ڿ� ����
				if (class_names[c] == "thirty")sound("thirty");     //Ŭ������ ���� �Լ� ȣ��
				else if (class_names[c] == "fifty")sound("fifty");     //Ŭ������ ���� �Լ� ȣ��
				else if (class_names[c] == "sixty")sound("sixty");     //Ŭ������ ���� �Լ� ȣ��
				else if (class_names[c] == "leftban")sound("leftban");     //Ŭ������ ���� �Լ� ȣ��
				else if (class_names[c] == "speedbump")sound("speedbump");     //Ŭ������ ���� �Լ� ȣ��
				else if (class_names[c] == "U_turn")sound("U_turn");     //Ŭ������ ���� �Լ� ȣ��
			}
		}
		imshow("output", frame);  // ���� ���
		waitKey();  // Ű���� ���
	}
	return 0;  //�Լ� ����
}
void sound(string name) {  // ���� �޼��� ����Լ� ����
	if (name == "thirty")PlaySound(TEXT("sound30"), 0, SND_FILENAME | SND_ASYNC);  //Ŭ������ ���� �����޼��� ���
	else if (name == "fifty")PlaySound(TEXT("sound50"), 0, SND_FILENAME | SND_ASYNC); //Ŭ������ ���� �����޼��� ���
	else if (name == "sixty")PlaySound(TEXT("sound60"), 0, SND_FILENAME | SND_ASYNC); //Ŭ������ ���� �����޼��� ���
	else if (name == "leftban")PlaySound(TEXT("soundleftban"), 0, SND_FILENAME | SND_ASYNC); //Ŭ������ ���� �����޼��� ���
	else if (name == "speedbump")PlaySound(TEXT("soundbump"), 0, SND_FILENAME | SND_ASYNC); //Ŭ������ ���� �����޼��� ���
	else if (name == "U_turn")PlaySound(TEXT("soundU_turn"), 0, SND_FILENAME | SND_ASYNC); //Ŭ������ ���� �����޼��� ���
}