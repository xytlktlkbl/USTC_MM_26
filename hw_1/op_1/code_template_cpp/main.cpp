// Copyright 2026, Yumeng Liu @ USTC
// op_1: Seam Carving — Student Template
//
// Deps  : OpenCV, STL
// Usage : ./op1_template [image_path]
//
// TODO: Implement seamCarveImage() below.

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// ============================================================
// TODO: Implement seamCarveImage
// ============================================================
//
// Seam carving to resize img to (target_rows, target_cols).
//
// You may define any helper functions above this function.
//
// Args:
//   img         — input BGR image (CV_8UC3)
//   target_rows — target height in pixels
//   target_cols — target width  in pixels
//
// Returns: resized image of size (target_rows, target_cols)
std::vector<std::vector<int>> compute_energy_map(const cv::Mat &img)
// compute energy with L1-gradient
{
    int rows = img.rows;
    int cols = img.cols;
    std::vector<std::vector<int>> energy(rows, std::vector<int>(cols, 0));
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (i == 0 && j == 0)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(0, 1) - img.at<cv::Vec3b>(0, 0), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(1, 0) - img.at<cv::Vec3b>(0, 0), cv::NORM_L1);
            }
            else if (i == 0 && j == cols - 1)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(0, cols - 2) - img.at<cv::Vec3b>(0, cols - 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(1, cols - 1) - img.at<cv::Vec3b>(0, cols - 1), cv::NORM_L1);
            }
            else if (i == rows - 1 && j == 0)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(rows - 2, 0) - img.at<cv::Vec3b>(rows - 1, 0), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(rows - 1, 1) - img.at<cv::Vec3b>(rows - 1, 0), cv::NORM_L1);
            }
            else if (i == rows - 1 && j == cols - 1)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(rows - 2, cols - 1) - img.at<cv::Vec3b>(rows - 1, cols - 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(rows - 1, cols - 2) - img.at<cv::Vec3b>(rows - 1, cols - 1), cv::NORM_L1);
            }
            else if (i == rows - 1)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j - 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j + 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i - 1, j), cv::NORM_L1);
            }
            else if (i == 0)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j - 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j + 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i + 1, j), cv::NORM_L1);
            }
            else if (j == cols - 1)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i - 1, j), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i + 1, j), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j - 1), cv::NORM_L1);
            }
            else if (j == 0)
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i - 1, j), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i + 1, j), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j + 1), cv::NORM_L1);
            }
            else
            {
                energy[i][j] = cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j - 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i, j + 1), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i - 1, j), cv::NORM_L1) + cv::norm(img.at<cv::Vec3b>(i, j) - img.at<cv::Vec3b>(i + 1, j), cv::NORM_L1);
            }
        }
    }
    return energy;
}

std::vector<int> dp_top_down(const std::vector<std::vector<int>> &energy)
{
    int rows = energy.size();
    int cols = energy[0].size();
    std::vector<std::vector<int>> dp(rows, std::vector<int>(cols, 0));
    for (int j = 0; j < cols; ++j)
    {
        dp[0][j] = energy[0][j];
    }
    for (int i = 1; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            dp[i][j] = dp[i - 1][j];
            if (j > 0)
                dp[i][j] = std::min(dp[i][j], dp[i - 1][j - 1]);
            if (j < cols - 1)
                dp[i][j] = std::min(dp[i][j], dp[i - 1][j + 1]);
            dp[i][j] += energy[i][j];
        }
    }
    std::vector<int> seam(rows);
    seam[rows - 1] = 0;
    for (int i = 0; i < cols; ++i)
    {
        if (dp[rows - 1][seam[rows - 1]] > dp[rows - 1][i])
        {
            seam[rows - 1] = i;
        }
    }
    for (int i = rows - 2; i >= 0; --i)
    {
        int j = seam[i + 1];
        seam[i] = j;
        if (j > 0 && dp[i][j - 1] < dp[i][seam[i]])
        {
            seam[i] = j - 1;
        }
        if (j < cols - 1 && dp[i][j + 1] < dp[i][seam[i]])
        {
            seam[i] = j + 1;
        }
    }
    return seam;
}

std::vector<std::vector<int>> dp_top_down(cv::Mat img, int k)
{
    int rows = img.rows;
    int cols = img.cols;

    std::vector<std::vector<int>> seams;
    std::vector<std::vector<int>> index_map(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            index_map[i][j] = j;

    for (int iter = 0; iter < k; ++iter)
    {
        std::vector<std::vector<int>> energy = compute_energy_map(img);
        std::vector<int> seam = dp_top_down(energy);
        std::vector<int> original_seam(rows);
        for (int i = 0; i < rows; ++i)
        {
            original_seam[i] = index_map[i][seam[i]];
        }

        seams.push_back(original_seam);
        cv::Mat new_img(rows, img.cols - 1, CV_8UC3);
        std::vector<std::vector<int>> new_map(rows, std::vector<int>(img.cols - 1));
        for (int i = 0; i < rows; ++i)
        {
            int col_to_remove = seam[i];
            int new_col = 0;
            for (int j = 0; j < img.cols; ++j)
            {
                if (j == col_to_remove)
                    continue;
                new_img.at<cv::Vec3b>(i, new_col) = img.at<cv::Vec3b>(i, j);
                new_map[i][new_col] = index_map[i][j];
                new_col++;
            }
        }
        img = new_img;
        index_map = new_map;
    }

    return seams;
}

std::vector<int> dp_left_right(const std::vector<std::vector<int>> &energy)
{
    int rows = energy.size();
    int cols = energy[0].size();
    std::vector<std::vector<int>> dp(rows, std::vector<int>(cols, 0));
    for (int i = 0; i < rows; ++i)
    {
        dp[i][0] = energy[i][0];
    }
    for (int j = 1; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            dp[i][j] = dp[i][j - 1];
            if (i > 0)
                dp[i][j] = std::min(dp[i][j], dp[i - 1][j - 1]);
            if (i < rows - 1)
                dp[i][j] = std::min(dp[i][j], dp[i + 1][j - 1]);
            dp[i][j] += energy[i][j];
        }
    }
    std::vector<int> seam(cols);
    seam[cols - 1] = 0;
    for (int i = 0; i < rows; ++i)
    {
        if (dp[seam[cols - 1]][cols - 1] > dp[i][cols - 1])
        {
            seam[cols - 1] = i;
        }
    }
    for (int j = cols - 2; j >= 0; --j)
    {
        int i = seam[j + 1];
        seam[j] = i;
        if (i > 0 && dp[i - 1][j] < dp[seam[j]][j])
        {
            seam[j] = i - 1;
        }
        if (i < rows - 1 && dp[i + 1][j] < dp[seam[j]][j])
        {
            seam[j] = i + 1;
        }
    }
    return seam;
}

std::vector<std::vector<int>> dp_left_right(cv::Mat img, int k)
{
    int rows = img.rows;
    int cols = img.cols;

    std::vector<std::vector<int>> seams;
    std::vector<std::vector<int>> index_map(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            index_map[i][j] = i;

    for (int iter = 0; iter < k; ++iter)
    {
        std::vector<std::vector<int>> energy = compute_energy_map(img);
        std::vector<int> seam = dp_left_right(energy);
        std::vector<int> original_seam(cols);
        for (int j = 0; j < cols; ++j)
        {
            original_seam[j] = index_map[seam[j]][j];
        }

        seams.push_back(original_seam);
        cv::Mat new_img(img.rows - 1, cols, CV_8UC3);
        std::vector<std::vector<int>> new_map(img.rows - 1, std::vector<int>(cols));
        for (int j = 0; j < cols; ++j)
        {
            int row_to_remove = seam[j];
            int new_row = 0;
            for (int i = 0; i < img.rows; ++i)
            {
                if (i == row_to_remove)
                    continue;
                new_img.at<cv::Vec3b>(new_row, j) = img.at<cv::Vec3b>(i, j);
                new_map[new_row][j] = index_map[i][j];
                new_row++;
            }
        }

        img = new_img;
        index_map = new_map;
    }

    return seams;
}

void remove_col_seam(cv::Mat &img, const std::vector<int> &seam)
{
    int rows = img.rows;
    int cols = img.cols;

    cv::Mat result(rows, cols - 1, CV_8UC3);

    for (int i = 0; i < rows; i++)
    {
        int col_to_remove = seam[i];

        int new_col = 0;
        for (int j = 0; j < cols; j++)
        {
            if (j == col_to_remove)
                continue;

            result.at<cv::Vec3b>(i, new_col++) = img.at<cv::Vec3b>(i, j);
        }
    }
    img = result;
}

void insert_col_seams(cv::Mat &img, const std::vector<std::vector<int>> &seams)
{
    int rows = img.rows;
    int cols = img.cols;
    int k = seams.size();

    cv::Mat result(rows, cols + k, CV_8UC3);

    for (int i = 0; i < rows; ++i)
    {
        std::vector<int> new_row;
        for (int s = 0; s < k; ++s)
        {
            new_row.push_back(seams[s][i]);
        }

        std::sort(new_row.begin(), new_row.end());

        int seam_idx = 0;
        int new_col = 0;

        for (int j = 0; j < cols; ++j)
        {
            result.at<cv::Vec3b>(i, new_col++) = img.at<cv::Vec3b>(i, j);

            while (seam_idx < k && new_row[seam_idx] == j)
            {
                cv::Vec3b new_pixel;
                if (j == 0)
                {
                    new_pixel = img.at<cv::Vec3b>(i, j + 1);
                }
                else if (j == cols - 1)
                {
                    new_pixel = img.at<cv::Vec3b>(i, j - 1);
                }
                else
                {
                    cv::Vec3b left = img.at<cv::Vec3b>(i, j - 1);
                    cv::Vec3b right = img.at<cv::Vec3b>(i, j + 1);
                    new_pixel = (left / 2 + right / 2);
                }

                result.at<cv::Vec3b>(i, new_col++) = new_pixel;
                seam_idx++;
            }
        }
    }

    img = result;
}

void remove_row_seam(cv::Mat &img, const std::vector<int> &seam)
{
    int rows = img.rows;
    int cols = img.cols;

    cv::Mat result(rows - 1, cols, CV_8UC3);

    for (int j = 0; j < cols; j++)
    {
        int row_to_remove = seam[j];

        int new_row = 0;
        for (int i = 0; i < rows; i++)
        {
            if (i == row_to_remove)
                continue;

            result.at<cv::Vec3b>(new_row++, j) = img.at<cv::Vec3b>(i, j);
        }
    }
    img = result;
}

void insert_row_seams(cv::Mat &img, const std::vector<std::vector<int>> &seams)
{
    int rows = img.rows;
    int cols = img.cols;
    int k = seams.size();

    cv::Mat result(rows + k, cols, CV_8UC3);

    for (int i = 0; i < cols; ++i)
    {
        std::vector<int> new_col;
        for (int s = 0; s < k; ++s)
        {
            new_col.push_back(seams[s][i]);
        }

        std::sort(new_col.begin(), new_col.end());

        int seam_idx = 0;
        int new_row = 0;

        for (int j = 0; j < rows; ++j)
        {
            result.at<cv::Vec3b>(new_row++, i) = img.at<cv::Vec3b>(j, i);

            while (seam_idx < k && new_col[seam_idx] == j)
            {
                cv::Vec3b new_pixel;
                if (j == 0)
                {
                    new_pixel = img.at<cv::Vec3b>(j + 1, i);
                }
                else if (j == rows - 1)
                {
                    new_pixel = img.at<cv::Vec3b>(j - 1, i);
                }
                else
                {
                    cv::Vec3b left = img.at<cv::Vec3b>(j - 1, i);
                    cv::Vec3b right = img.at<cv::Vec3b>(j + 1, i);
                    new_pixel = (left / 2 + right / 2);
                }

                result.at<cv::Vec3b>(new_row++, i) = new_pixel;
                seam_idx++;
            }
        }
    }

    img = result;
}

cv::Mat seamCarveImage(cv::Mat img, int target_rows, int target_cols)
{
    // TODO: replace with your implementation
    if (img.cols > target_cols)
    {
        while (img.cols > target_cols)
        {
            std::vector<std::vector<int>> energy = compute_energy_map(img);
            std::vector<int> seam = dp_top_down(energy);
            remove_col_seam(img, seam);
        }
    }

    else if (img.cols < target_cols)
    {
        int k = target_cols - img.cols;
        std::vector<std::vector<int>> seam = dp_top_down(img, k);
        insert_col_seams(img, seam);
    }
    if (img.rows > target_rows)
    {
        while (img.rows > target_rows)
        {
            std::vector<std::vector<int>> energy = compute_energy_map(img);
            std::vector<int> seam = dp_left_right(energy);
            remove_row_seam(img, seam);
        }
    }
    else if (img.rows < target_rows)
    {
        int k = target_rows - img.rows;
        std::vector<std::vector<int>> seam = dp_left_right(img, k);
        insert_row_seams(img, seam);
    }
    return img;
}

cv::Mat truncate(cv::Mat img, int target_rows, int target_cols)
{
    int rows = img.rows;
    int cols = img.cols;
    if (target_rows > rows || target_cols > cols)
    {
        return img;
    }
    cv::Mat result(target_rows, target_cols, CV_8UC3);
    for (int i = 0; i < target_rows; ++i)
    {
        for (int j = 0; j < target_cols; ++j)
        {
            result.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(i, j);
        }
    }
    return result;
}

cv::Mat resize_fig(cv::Mat img, int target_rows, int target_cols)
{
    float row_scale = float(target_rows) / img.rows;
    float col_scale = float(target_cols) / img.cols;
    cv::Mat result(target_rows, target_cols, CV_8UC3);
    for (int i = 0; i < target_rows; ++i)
    {
        for (int j = 0; j < target_cols; ++j)
        {
            int src_i = std::min(int(i / row_scale), img.rows - 1);
            int src_j = std::min(int(j / col_scale), img.cols - 1);
            result.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(src_i, src_j);
        }
    }
    return result;
}

std::vector<int> dp_top_down_forward(const cv::Mat &img)
{
    int rows = img.rows;
    int cols = img.cols;

    std::vector<std::vector<int>> dp(rows, std::vector<int>(cols, 0));
    std::vector<std::vector<int>> path(rows, std::vector<int>(cols, 0));

    for (int j = 0; j < cols; ++j)
    {
        int left = std::max(j - 1, 0);
        int right = std::min(j + 1, cols - 1);
        dp[0][j] = cv::norm(img.at<cv::Vec3b>(0, right) - img.at<cv::Vec3b>(0, left), cv::NORM_L1);
    }

    for (int i = 1; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            int left = std::max(j - 1, 0);
            int right = std::min(j + 1, cols - 1);

            cv::Vec3b p_up = img.at<cv::Vec3b>(i - 1, j);
            cv::Vec3b p_left = img.at<cv::Vec3b>(i, left);
            cv::Vec3b p_right = img.at<cv::Vec3b>(i, right);
            int cU = cv::norm(p_right - p_left, cv::NORM_L1);
            int cL = cU + cv::norm(p_up - p_left, cv::NORM_L1);
            int cR = cU + cv::norm(p_up - p_right, cv::NORM_L1);
            int cost_U = dp[i - 1][j] + cU;
            int cost_L = (j > 0) ? (dp[i - 1][j - 1] + cL) : INT_MAX;
            int cost_R = (j < cols - 1) ? (dp[i - 1][j + 1] + cR) : INT_MAX;

            if (cost_U <= cost_L && cost_U <= cost_R)
            {
                dp[i][j] = cost_U;
                path[i][j] = 0;
            }
            else if (cost_L <= cost_U && cost_L <= cost_R)
            {
                dp[i][j] = cost_L;
                path[i][j] = -1;
            }
            else
            {
                dp[i][j] = cost_R;
                path[i][j] = 1;
            }
        }
    }

    std::vector<int> seam(rows);
    int min_val = INT_MAX;
    int min_idx = 0;
    for (int j = 0; j < cols; ++j)
    {
        if (dp[rows - 1][j] < min_val)
        {
            min_val = dp[rows - 1][j];
            min_idx = j;
        }
    }

    seam[rows - 1] = min_idx;
    for (int i = rows - 1; i > 0; --i)
    {
        int dir = path[i][min_idx];
        min_idx += dir;
        seam[i - 1] = min_idx;
    }

    return seam;
}

std::vector<std::vector<int>> dp_top_down_forward(cv::Mat img, int k)
{
    int rows = img.rows;
    int cols = img.cols;

    std::vector<std::vector<int>> seams;
    std::vector<std::vector<int>> index_map(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            index_map[i][j] = j;

    for (int iter = 0; iter < k; ++iter)
    {
        std::vector<int> seam = dp_top_down_forward(img);
        std::vector<int> original_seam(rows);
        for (int i = 0; i < rows; ++i)
        {
            original_seam[i] = index_map[i][seam[i]];
        }

        seams.push_back(original_seam);
        cv::Mat new_img(rows, img.cols - 1, CV_8UC3);
        std::vector<std::vector<int>> new_map(rows, std::vector<int>(img.cols - 1));
        for (int i = 0; i < rows; ++i)
        {
            int col_to_remove = seam[i];
            int new_col = 0;
            for (int j = 0; j < img.cols; ++j)
            {
                if (j == col_to_remove)
                    continue;
                new_img.at<cv::Vec3b>(i, new_col) = img.at<cv::Vec3b>(i, j);
                new_map[i][new_col] = index_map[i][j];
                new_col++;
            }
        }
        img = new_img;
        index_map = new_map;
    }

    return seams;
}

std::vector<int> dp_left_right_forward(const cv::Mat &img)
{
    int rows = img.rows;
    int cols = img.cols;

    std::vector<std::vector<int>> dp(rows, std::vector<int>(cols, 0));
    std::vector<std::vector<int>> path(rows, std::vector<int>(cols, 0));

    for (int i = 0; i < rows; ++i)
    {
        int top = std::max(i - 1, 0);
        int bottom = std::min(i + 1, rows - 1);
        dp[i][0] = cv::norm(img.at<cv::Vec3b>(bottom, 0) - img.at<cv::Vec3b>(top, 0), cv::NORM_L1);
    }

    for (int j = 1; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            int top = std::max(i - 1, 0);
            int bottom = std::min(i + 1, rows - 1);

            cv::Vec3b p_left = img.at<cv::Vec3b>(i, j - 1);
            cv::Vec3b p_top = img.at<cv::Vec3b>(top, j);
            cv::Vec3b p_bottom = img.at<cv::Vec3b>(bottom, j);
            int cS = cv::norm(p_bottom - p_top, cv::NORM_L1);
            int cUL = cS + cv::norm(p_left - p_bottom, cv::NORM_L1);
            int cDL = cS + cv::norm(p_left - p_top, cv::NORM_L1);

            int cost_S = dp[i][j - 1] + cS;
            int cost_UL = (i > 0) ? (dp[i - 1][j - 1] + cUL) : INT_MAX;
            int cost_DL = (i < rows - 1) ? (dp[i + 1][j - 1] + cDL) : INT_MAX;

            if (cost_S <= cost_UL && cost_S <= cost_DL)
            {
                dp[i][j] = cost_S;
                path[i][j] = 0;
            }
            else if (cost_UL <= cost_S && cost_UL <= cost_DL)
            {
                dp[i][j] = cost_UL;
                path[i][j] = -1;
            }
            else
            {
                dp[i][j] = cost_DL;
                path[i][j] = 1;
            }
        }
    }

    std::vector<int> seam(cols);
    int min_val = INT_MAX;
    int min_idx = 0;

    for (int i = 0; i < rows; ++i)
    {
        if (dp[i][cols - 1] < min_val)
        {
            min_val = dp[i][cols - 1];
            min_idx = i;
        }
    }

    seam[cols - 1] = min_idx;
    for (int j = cols - 1; j > 0; --j)
    {
        int dir = path[min_idx][j];
        min_idx += dir;
        seam[j - 1] = min_idx;
    }

    return seam;
}

std::vector<std::vector<int>> dp_left_right_forward(cv::Mat img, int k)
{
    int rows = img.rows;
    int cols = img.cols;

    std::vector<std::vector<int>> seams;
    std::vector<std::vector<int>> index_map(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            index_map[i][j] = i;

    for (int iter = 0; iter < k; ++iter)
    {
        std::vector<int> seam = dp_left_right_forward(img);
        std::vector<int> original_seam(cols);
        for (int j = 0; j < cols; ++j)
        {
            original_seam[j] = index_map[seam[j]][j];
        }

        seams.push_back(original_seam);
        cv::Mat new_img(img.rows - 1, cols, CV_8UC3);
        std::vector<std::vector<int>> new_map(img.rows - 1, std::vector<int>(cols));
        for (int j = 0; j < cols; ++j)
        {
            int row_to_remove = seam[j];
            int new_row = 0;
            for (int i = 0; i < img.rows; ++i)
            {
                if (i == row_to_remove)
                    continue;
                new_img.at<cv::Vec3b>(new_row, j) = img.at<cv::Vec3b>(i, j);
                new_map[new_row][j] = index_map[i][j];
                new_row++;
            }
        }

        img = new_img;
        index_map = new_map;
    }

    return seams;
}

cv::Mat forward_energy(cv::Mat img, int target_rows, int target_cols)
{
    if (img.cols > target_cols)
    {
        int n = 0;
        while (img.cols > target_cols)
        {
            std::vector<int> seam = dp_top_down_forward(img);
            remove_col_seam(img, seam);
        }
    }

    else if (img.cols < target_cols)
    {
        int k = target_cols - img.cols;
        std::vector<std::vector<int>> seam = dp_top_down_forward(img, k);
        insert_col_seams(img, seam);
    }
    if (img.rows > target_rows)
    {
        int n = 0;
        while (img.rows > target_rows)
        {
            std::vector<int> seam = dp_left_right_forward(img);
            remove_row_seam(img, seam);
        }
    }
    else if (img.rows < target_rows)
    {
        int k = target_rows - img.rows;
        std::vector<std::vector<int>> seam = dp_left_right(img, k);
        insert_row_seams(img, seam);
    }
    return img;
}

// ============================================================
// GUI
// ============================================================

static cv::Mat g_src, g_dst;
static int g_col_pct = 100, g_row_pct = 100;
// === 新增：算法模式选择 ===
enum AlgoMode
{
    MODE_TRUNCATE = 0,
    MODE_STRETCH,
    MODE_SEAM_CARVING,
    MODE_FORWARD
};
static int g_algo_mode = MODE_SEAM_CARVING; // 默认选中 Seam Carving
const char *g_algo_labels[] = {"Truncate", "Stretch", "SeamCarving", "Forward"};

static void refresh()
{
    const int PAD = 20;
    const int HDR = 70; // 【修改点1】把高度从 36 改到 70，留出空间显示模式

    int h = g_src.rows + HDR;
    int w = g_src.cols;
    if (!g_dst.empty())
    {
        h = std::max(g_src.rows, g_dst.rows) + HDR;
        w = g_src.cols + g_dst.cols + PAD;
    }

    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(45, 45, 45));

    cv::rectangle(canvas, cv::Point(0, 0), cv::Point(canvas.cols, 35), cv::Scalar(30, 30, 30), -1);

    std::string mode_name = g_algo_labels[g_algo_mode];

    cv::putText(canvas, "Current Mode: " + mode_name, cv::Point(10, 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 100), 2, cv::LINE_AA);

    g_src.copyTo(canvas(cv::Rect(0, HDR, g_src.cols, g_src.rows)));

    cv::putText(canvas,
                "Input [" + std::to_string(g_src.cols) + " x " + std::to_string(g_src.rows) + "]",
                cv::Point(4, HDR - 10), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(210, 210, 210), 1, cv::LINE_AA);

    if (!g_dst.empty())
    {
        int xoff = g_src.cols + PAD;
        g_dst.copyTo(canvas(cv::Rect(xoff, HDR, g_dst.cols, g_dst.rows)));
        cv::putText(canvas,
                    "Result [" + std::to_string(g_dst.cols) + " x " + std::to_string(g_dst.rows) + "]",
                    cv::Point(xoff + 4, HDR - 10), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(210, 210, 210), 1, cv::LINE_AA);
    }
    else
    {
        cv::putText(canvas, "Adjust sliders then press [Space] to process",
                    cv::Point(10, h / 2 + 10), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(80, 220, 80), 2, cv::LINE_AA);
    }
    cv::imshow("Seam Carving", canvas);
}

int main(int argc, char *argv[])
{
    std::string path;
    if (argc > 1)
    {
        path = argv[1];
        g_src = cv::imread(path, cv::IMREAD_COLOR);
    }
    else
    {
        for (const char *p : {"../figs/original.png", "../../figs/original.png"})
        {
            g_src = cv::imread(p, cv::IMREAD_COLOR);
            if (!g_src.empty())
            {
                path = p;
                break;
            }
        }
        if (g_src.empty())
            path = "../figs/original.png";
    }
    if (g_src.empty())
    {
        std::cerr << "Cannot open image: " << path << "\n"
                  << "Usage: op1_template [image_path]\n";
        return 1;
    }

    std::cout << "Image: " << g_src.cols << " x " << g_src.rows << " px\n"
              << "Keys : [Space] run  [s] save  [r] reset  [q/Esc] quit\n";

    cv::namedWindow("Seam Carving", cv::WINDOW_NORMAL);
    int win_w = std::min(g_src.cols * 2 + 140, 1600);
    cv::resizeWindow("Seam Carving", win_w, g_src.rows + 120);

    cv::createTrackbar("Col %", "Seam Carving", &g_col_pct, 200);
    cv::createTrackbar("Row %", "Seam Carving", &g_row_pct, 200);
    cv::createTrackbar("Algorithm", "Seam Carving", &g_algo_mode, 3, nullptr);
    cv::setTrackbarPos("Col %", "Seam Carving", 100);
    cv::setTrackbarPos("Row %", "Seam Carving", 100);
    cv::setTrackbarPos("Algorithm", "Seam Carving", MODE_SEAM_CARVING);

    refresh();

    while (true)
    {
        int key = cv::waitKey(30) & 0xFF;
        if (key == 27 || key == 'q')
            break;

        if (key == ' ')
        {
            int col_pct = std::max(10, g_col_pct);
            int row_pct = std::max(10, g_row_pct);
            int tgt_w = std::max(1, g_src.cols * col_pct / 100);
            int tgt_h = std::max(1, g_src.rows * row_pct / 100);
            std::cout << "Running: " << g_src.cols << "x" << g_src.rows
                      << " -> " << tgt_w << "x" << tgt_h << " ...\n";
            switch (g_algo_mode)
            {
            case 0:
                g_dst = truncate(g_src.clone(), tgt_h, tgt_w);
                break;
            case 1:
                g_dst = resize_fig(g_src.clone(), tgt_h, tgt_w);
                break;
            case 3:
                g_dst = forward_energy(g_src.clone(), tgt_h, tgt_w);
                break;
            case 2:
                g_dst = seamCarveImage(g_src.clone(), tgt_h, tgt_w);
                break;
            }

            std::cout << "Done.\n";
            refresh();
        }

        if (key == 'r')
        {
            g_dst = cv::Mat();
            cv::setTrackbarPos("Col %", "Seam Carving", 100);
            cv::setTrackbarPos("Row %", "Seam Carving", 100);
            refresh();
        }

        if (key == 's' && !g_dst.empty())
        {
            cv::imwrite("result.png", g_dst);
            std::cout << "Saved result.png\n";
        }
    }
    return 0;
}
