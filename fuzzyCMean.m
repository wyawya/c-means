clc
clear
close all
img = double(imread('C:\Users\zhangbohang\Desktop\temp.jpg'));
subplot(1,2,1),imshow(img,[]);
data = img(:);
%�ֳ�4��
[center,U,obj_fcn] = fcm(data,2);
[~,label] = max(U); %�ҵ���������
%�仯��ͼ��Ĵ�С
img_new = reshape(label,size(img));
subplot(1,2,2),imshow(img_new,[]);