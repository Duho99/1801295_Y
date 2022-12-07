#include <iostream>     //<iostream>포함
#include <opencv2/opencv.hpp>     //<opencv2/opencv.hpp>포함
#include <stdio.h>     //<stdio.h>포함
#include <Windows.h>     //<Windows.h>포함
#include <conio.h>     //<conio.h>포함
#include <mmsystem.h>     //<mmsystem.h>포함
#pragma comment(lib,"winmm.lib")  //winmm.lib 라이브러리 추간
using namespace std;  //std 생략
using namespace cv;  //cv 생략
using namespace cv::dnn;  //dnn 모듈 포함
const int NUM_CLASSES = 7;  //클래스 개수
const Scalar colors[] = {{0, 255, 255},{255, 255, 0},{0, 255, 0},{255, 0, 0}};  //객체 배열 선언
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);  //컬러 번호
void sound(string name);  // 음성 메세지 출력함수 선언
int main() {  //메인함수
	vector <string> class_names = { "thirty", "fifty", "sixty", "speedbump", "leftban", "U_turn", "car" };  //클래스 이름 벡터
	auto net = cv::dnn::readNetFromDarknet("yolov4-traffic.cfg", "yolov4-traffic_final.weights");  //미리 학습된 모델로 Net객체 생성
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV); // 계산에 opencv를 사용하도록함
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);  // 계산에 CPU를 사용하도록함
	auto output_names = net.getUnconnectedOutLayersNames(); //연결되지 않은 레이어의 이름을 반환
	for (int i = 1; i <= 7; i++) {  //반복문
		string filename = format("test%d.jpg", i);   // 파일이름 저장
		Mat frame = imread(filename);  // 테스트 데이터 읽기
		Mat blob;  // blob객체 생성
		vector<Mat> detections;  //디텍션 객체 생성
		if (frame.empty()){		// 테스트 데이터 유무 조건문	
			cerr << "frame empty" << endl; return -1;}  //에러 메세지 출력
		cv::dnn::blobFromImage(frame, blob, 1 / 255.f, Size(320,320), Scalar(), true, false, CV_32F);  //입력영상으로 부터 4차원 블롭 객체 반환
		net.setInput(blob);  //블롭 객체를 네트워크 입력으로 설정
		TickMeter tm;   // 객체 선언
		tm.start();  //시간 측정 시작
		net.forward(detections, output_names);  //추론하여 결과를 객체로 반환
		tm.stop();  //시간 측정 마감
		cout << "time : " << tm.getTimeMilli() << endl;  //추론 시간 출력
		vector<int> indices[NUM_CLASSES];  // 클래스 인덱스
		vector<cv::Rect> boxes[NUM_CLASSES];  //클래스 바운딩 박스
		vector<float> scores[NUM_CLASSES];   //클래스 신뢰도
		for (auto& output : detections)  //반복문
		{
			const auto num_boxes = output.rows; // 추론된 객체 행
			for (int i = 0; i < num_boxes; i++)  // 이중 반복문
			{
				auto x = output.at<float>(i, 0) * frame.cols;   //바운딩 박스 중심좌표
				auto y = output.at<float>(i, 1) * frame.rows;   //바운딩 박스 중심좌표
				auto width = output.at<float>(i, 2) * frame.cols;   //바운딩 박스 폭
				auto height = output.at<float>(i, 3) * frame.rows;   //바운딩 박스 높이
				Rect rect(x - width / 2, y - height / 2, width, height);  //바운딩 박스 Rect 객체
				for (int c = 0; c < NUM_CLASSES; c++)  //class X 반복문
				{
					auto confidence = *output.ptr<float>(i, 5 + c);   //바운딩 박스안에 class X 객체가 존재할 확률
					if (confidence >= 0.5)  // 확률 신뢰도가 0.5보다 크면
					{
						boxes[c].push_back(rect); //바운딩 박스 정보 저장
						scores[c].push_back(confidence);  //신뢰도 저장
					}
				}
			}
		}
		for (int c = 0; c < NUM_CLASSES; c++)    //반복문
			cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, 0.5, indices[c]);   //주어진 상자와 해당 점수가 아닌 최대 억제를 수행
		for (int c = 0; c < NUM_CLASSES; c++)    //반복문
		{
			for (int i = 0; i < indices[c].size(); ++i)  //반복문
			{
				const auto color = colors[c % NUM_COLORS];  //컬러지정
				auto idx = indices[c][i];  //인덱스
				const auto& rect = boxes[c][idx];  //바운딩 박스
				rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), color, 3);  //사각형 그리기 (바운딩 박스)
				string label_str = class_names[c] + ": " + format("%.02lf", scores[c][idx]);  //클래스명과 신뢰도 문자열
				int baseline; //변수선언
				auto label_bg_sz = getTextSize(label_str, FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);  //label_str 텍스트 사이즈
				rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), 
					Point(rect.x + label_bg_sz.width, rect.y), color, FILLED);  //사각형 그리기 (클래스 이름, 신뢰도)
				putText(frame, label_str, Point(rect.x, rect.y - baseline - 5),
					FONT_HERSHEY_COMPLEX_SMALL, 1,Scalar(0, 0, 0));  //클래스명과 신뢰도 문자열 쓰기
				if (class_names[c] == "thirty")sound("thirty");     //클래스명에 따른 함수 호출
				else if (class_names[c] == "fifty")sound("fifty");     //클래스명에 따른 함수 호출
				else if (class_names[c] == "sixty")sound("sixty");     //클래스명에 따른 함수 호출
				else if (class_names[c] == "leftban")sound("leftban");     //클래스명에 따른 함수 호출
				else if (class_names[c] == "speedbump")sound("speedbump");     //클래스명에 따른 함수 호출
				else if (class_names[c] == "U_turn")sound("U_turn");     //클래스명에 따른 함수 호출
			}
		}
		imshow("output", frame);  // 영상 출력
		waitKey();  // 키보드 대기
	}
	return 0;  //함수 종료
}
void sound(string name) {  // 음성 메세지 출력함수 정의
	if (name == "thirty")PlaySound(TEXT("sound30"), 0, SND_FILENAME | SND_ASYNC);  //클래스명에 따른 음성메세지 출력
	else if (name == "fifty")PlaySound(TEXT("sound50"), 0, SND_FILENAME | SND_ASYNC); //클래스명에 따른 음성메세지 출력
	else if (name == "sixty")PlaySound(TEXT("sound60"), 0, SND_FILENAME | SND_ASYNC); //클래스명에 따른 음성메세지 출력
	else if (name == "leftban")PlaySound(TEXT("soundleftban"), 0, SND_FILENAME | SND_ASYNC); //클래스명에 따른 음성메세지 출력
	else if (name == "speedbump")PlaySound(TEXT("soundbump"), 0, SND_FILENAME | SND_ASYNC); //클래스명에 따른 음성메세지 출력
	else if (name == "U_turn")PlaySound(TEXT("soundU_turn"), 0, SND_FILENAME | SND_ASYNC); //클래스명에 따른 음성메세지 출력
}
