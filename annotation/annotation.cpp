#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <dirent.h>
#include <vector>
#include <string>
#include <fstream>
#include <random>

cv::Rect box;
bool drawing_box = false;
        std::vector<cv::Rect> v_roi;

void draw_box(cv::Mat *img, cv::Rect rect)
{
    cv::rectangle(*img, cv::Point2d(box.x, box.y), cv::Point2d(box.x + box.width, box.y + box.height),
                  cv::Scalar(0xff, 0x00, 0x00));
}

void my_mouse_callback(int event, int x, int y, int flags, void *param)
{
    cv::Mat *image = static_cast<cv::Mat *>(param);

    switch (event)
    {
    case cv::EVENT_MOUSEMOVE:
        if (drawing_box)
        {
            box.width = x - box.x;
            box.height = y - box.y;
        }
        break;

    case cv::EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = cv::Rect(x, y, 0, 0);
        break;

    case cv::EVENT_LBUTTONUP:
        drawing_box = false;
        if (box.width < 0)
        {
            box.x += box.width;
            box.width *= -1;
        }
        if (box.height < 0)
        {
            box.y += box.height;
            box.height *= -1;
        }
        box.height = std::min(image->size().height, box.height);
        box.width = std::min(image->size().width, box.width);
        v_roi.push_back(box);
        draw_box(image, box);
        break;
    }
}

int main()
{
    printf("Hello\n");
std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
   std::uniform_real_distribution<> distribution(0.0, 1.2);
    const std::string src_image_path = "/home/yukihiro/workspace/yolo/annotation/data/source/image/";
    const std::string target_image_path = "/home/yukihiro/workspace/yolo/annotation/data/target/image/";
    const std::string target_labeled_image_path = "/home/yukihiro/workspace/yolo/annotation/data/target/labeled_image/";
    const std::string target_label_path = "/home/yukihiro/workspace/yolo/annotation/data/target/label/";
    std::vector<std::string> v_source_image_file_name;
    {
        // all file
        DIR *dp;       // ディレクトリへのポインタ
        dirent *entry; // readdir() で返されるエントリーポイント
        dp = opendir(src_image_path.c_str());
        if (dp == NULL)
            exit(1);
        do
        {
            entry = readdir(dp);
            if (entry != NULL)
            {
                if (std::string(entry->d_name).find(".jpg") != std::string::npos)
                    v_source_image_file_name.push_back(entry->d_name);
            }
        } while (entry != NULL);
    }
    std::string window_name = "display";
    cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    for (const auto &source_image_file_name : v_source_image_file_name)
    {
    START:
        std::cout << source_image_file_name << std::endl;
        cv::Mat image = cv::imread(src_image_path + source_image_file_name);
        cv::Mat rendered_image = image.clone();
        cv::setMouseCallback(window_name, my_mouse_callback, (void *)&rendered_image);
        cv::Mat rendering_image = image.clone();
        v_roi.clear();
        bool skip = false;
        bool start_over = false;
        while (1)
        {
            rendered_image.copyTo(rendering_image);
            if (drawing_box)
            {
                draw_box(&rendering_image, box);
            }
            cv::imshow(window_name, rendering_image);
            int ret = cv::waitKey(15);
            if (ret == 10) //enter
                break;
            if (ret == 32) //space
            {
                skip = true;
                break;
            }
            if (ret == 27) //esc
            {
                start_over = true;
                break;
            }
        }

        if (skip){
                     std::cout << "skip" << std::endl;
            continue;
        }
        if (start_over)
        {
            std::cout << "start over" << std::endl;
            goto START;
        }
        // std::cout << "reset" << std::endl;
        // cv::imshow(window_name, image);
        // cv::waitKey(0);
        for (size_t i = 0; i < v_roi.size(); ++i)
        {
            int clipped_x, clipped_y, clipped_h, clipped_w;
            clipped_x = std::max(int(v_roi.at(i).x - v_roi.at(i).width * distribution(engine)), 0);
            clipped_y = std::max(int(v_roi.at(i).y - v_roi.at(i).height* distribution(engine)), 0);
            clipped_w = std::min(int((v_roi.at(i).x - clipped_x) + v_roi.at(i).width  + v_roi.at(i).width * distribution(engine)), image.size().width - clipped_x - 1);
            clipped_h = std::min(int((v_roi.at(i).y - clipped_y) + v_roi.at(i).height + v_roi.at(i).height * distribution(engine)), image.size().height - clipped_y - 1);
            std::cout << clipped_x << std::endl;
            std::cout << clipped_y << std::endl;
            std::cout << clipped_w <<", " << (v_roi.at(i).x - clipped_x)<< std::endl;
            std::cout << clipped_h <<", " << (v_roi.at(i).y - clipped_y)<< std::endl;
            std::cout << image.size().width  << std::endl;
            std::cout <<  image.size().height << std::endl;
            cv::Mat clipped = image(cv::Rect(cv::Point(clipped_x,clipped_y), cv::Size(clipped_w, clipped_h)));
            // cv::imshow(window_name, clipped);
            // cv::waitKey(0);

            std::ofstream fd_label_file;
            std::string target_label_file_name = std::string(source_image_file_name).erase(source_image_file_name.find(".jpg"), 4) + "_" + std::to_string(i) + ".txt";
            std::string target_image_file_name = std::string(source_image_file_name).erase(source_image_file_name.find(".jpg"), 4) + "_" + std::to_string(i) + ".jpg";
            fd_label_file.open(target_label_path + target_label_file_name);
            double norm_x, norm_y, norm_h, norm_w;
            norm_x = ((double)(v_roi.at(i).x - clipped_x) + (double)(v_roi.at(i).width) / 2.0) / (double)clipped.size().width;
            norm_y = ((double)(v_roi.at(i).y - clipped_y) + (double)(v_roi.at(i).height) / 2.0) / (double)clipped.size().height;
            norm_w = ((double)v_roi.at(i).width) / (double)clipped.size().width;
            norm_h = ((double)v_roi.at(i).height) / (double)clipped.size().height;
            fd_label_file << std::string("0") + std::string(" ") + std::to_string(norm_x) + " " + std::to_string(norm_y) + " " + std::to_string(norm_w) + " " + std::to_string(norm_h) << std::endl;
            cv::imwrite(target_image_path + target_image_file_name, clipped);

            cv::Mat rendering_image = clipped.clone();
            cv::rectangle(rendering_image,
                          cv::Point2d((v_roi.at(i).x - clipped_x), (v_roi.at(i).y - clipped_y)),
                          cv::Point2d((v_roi.at(i).x - clipped_x) + v_roi.at(i).width, (v_roi.at(i).y - clipped_y) + v_roi.at(i).height),
                          cv::Scalar(0xff, 0x00, 0x00));

            cv::imwrite(target_labeled_image_path + target_image_file_name, rendering_image);
            // cv::imshow(window_name, rendering_image);
            // cv::waitKey(0);

            std::cout << target_label_path + target_label_file_name << std::endl;
            std::cout << target_image_path + target_image_file_name << std::endl;
        }
    }

    return 0;
}