 import matplotlib.pyplot as plt
 import numpy as np
 import pandas as pd
 import math
 from sklearn import datasets, linear_model
 from sklearn.metrics import mean_squared_error, r2_score

# lay du lieu diabetes- du lieu ve benh tieu duong
 diabetes = datasets.load_diabetes()
 print("Số chiều dữ liệu input: ", diabetes.data.shape)
 print("Kiểu dữ liệu input: ", type(diabetes.data))
 print("Số chiều dữ liệu target: ", diabetes.target.shape)
 print("Kiểu dữ liệu target: ", type(diabetes.target))
 print()
 print("5 mẫu dữ liệu đầu tiên:")
 print("input: ", diabetes.data[:5])
 print("target: ",diabetes.target[:5])
 #print("data[5,1]", diabetes.data[4,1])

  # cat nho du lieu, lay 1 phan cho qua trinh thu nghiem,
 # chia train test cac mau du lieu
 # diabetes_X = diabetes.data[:, np.newaxis, 2]
 diabetes_X = diabetes.data
 diabetes_X_train = diabetes_X[:361]
 diabetes_y_train = diabetes.target[:361]
 diabetes_X_test = diabetes_X[362:]
 diabetes_y_test = diabetes.target[362:]

# Xay dung model su dung sklearn
 regr = linear_model.LinearRegression()

 ##### exercise #####
 # Yêu cầu: Cài đặt mô hình Ridge Regression với alpha = 0.1
 # Gợi ý: xem hướng dẫn tại https://scikit-learn.org/stable/modules/generated/
 #↪sklearn.linear_model.Ridge.html
 ######################

 # Huấn luyện mô hình Linear Regression
 regr.fit(diabetes_X_train, diabetes_y_train)
 print("[w1, ... w_n] = ", regr.coef_)
 print("w0 = ", regr.intercept_)

 ##### exercise #####
 # Yêu cầu: Huấn luyện mô hình Ridge Regression và in ra các trọng số w0, w1, ...
 #↪,wn của mô hình
 # Gợi ý: xem hướng dẫn tại https://scikit-learn.org/stable/modules/generated/
 #↪sklearn.linear_model.Ridge.html
 ######################

 ##### exercise #####
 # Yêu cầu: tính giá trị dự đoán của mô hình trên mẫu đầu tiên của tập test và␣
 #↪so sánh với kết quả của thư viện
 # Gợi ý: sử dụng công thức y = w0 + w1*x1 + w1*x2 + ... + w_n*x_n
 ######################
 #Dự đoán thử cho trường hợp đầu tiên
 #Giá trị đúng
 print("Gia tri true: ", diabetes_y_test[0])
 #Dự đoán cho mô hình Linear Regression sử dụng hàm dự đoán của thư viện
 y_pred_linear = regr.predict(diabetes_X_test[0:1])
 print("Gia tri du doan cho mô hình linear regression: ", y_pred_linear)

#Viết code tính và in kết quả dự đoán cho mô hình Linear Regression sử dụng␣
 #↪công thức tại đây
 #Dự đoán cho mô hình Ridge Regression sử dụng hàm dự đoán của thư viện
 y_pred_ridge = regr_ridge.predict(diabetes_X_test[0:1])
 print("Gia tri du doan cho mô hình ridge regression: ", y_pred_ridge)
 #Viết code tính và in kết quả dự đoán cho mô hình Ridge Regression sử dụng công␣
# ↪thức tại đây
 ######################

# Thực hiện suy diễn sau khi huấn luyện
 diabetes_y_pred = regr.predict(diabetes_X_test)
 pd.DataFrame(data=np.array([diabetes_y_test, diabetes_y_pred,
 abs(diabetes_y_test- diabetes_y_pred)]).T,
 columns=["Thực tế", "Dự đoán", "Lệch"])

 # pd.DataFrame(data=np.array([diabetes_y_test, diabetes_y_pred,
 # abs(diabetes_y_test- diabetes_y_pred)]),
 # index=["Thực tế", "Dự đoán", "Lệch"])

# Giá trị RMSE của mô hình Linear Regression
 math.sqrt(mean_squared_error(diabetes_y_test, diabetes_y_pred))

 ##### exercise #####
 # Yêu cầu: đánh giá độ đo RMSE của mô hình Ridge Regression với các hằng số␣
 ##↪phạt khác nhau, in ra kết quả.
 # Gợi ý: Các bước làm:
 #- Lặp theo danh sách các hằng số phạt
 #- Dựng các mô hình Ridge Regression với mỗi hằng số phạt tương ứng
 #- Huấn luyện các mô hình và dự đoán
 #- Tính RMSE tương ứng
 ######################
 #Các giá trị hằng số phạt cho trước
 _lambda = [0, 0.0001,0.01, 0.04, 0.05, 0.06, 0.1, 0.5, 1, 5, 10, 20]

 import seaborn as sns
 sns.distplot(diabetes_y_test)
 pd.DataFrame(data=diabetes_y_test, columns=["values"]).describe()

 ##### exercise #####
 # Yêu cầu: Tính các chỉ số thống kê và vẽ biểu đồ phân phối của chỉ số dự đoán␣
 # ↪bằng mô hình Linear Regression, quan sát và nhận xét
 # Gợi ý: sử dụng sns và pd
 ######################

 import matplotlib.pyplot as plt
 plt.figure(figsize=(12,8))
 plt.plot(diabetes_y_test)
 plt.plot(diabetes_y_pred)
 plt.xlabel('Patients')
 plt.ylabel('Index')
 # function to show the plot
 plt.show()