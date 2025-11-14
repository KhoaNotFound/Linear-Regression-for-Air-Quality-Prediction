#Lưu ý các mảng trong đây được xây dựng trên numpy array nên việc viết code phép toán trực tiếp trên mảng sẽ trả về giá trị mảng mới với từng giá trị trong mảng là kết quả tính toán của từng giá trị mảng đầu vào tương ứng, đây là tính chất cốt lõi khiến python rất mạnh trong lĩnh vực phân tích dữ liệu
#--------------------------------
#CÁC KIẾN THỨC TOÁN HỌC CẦN THIẾT
#--------------------------------
#1.Thống kê:
#   Các kiến thức thống kê cơ bản (mean, variance,...)
#   Chuẩn hoá z_score (standardization)
#       z = (x - mu)/sigma để co giãn dữ liệu X_Train. Đây là 1 kỹ thuật thống kê cơ bản để đưa các biến về cùng 1 thang đo mà không làm thay đổi y. Cách chứng minh công thức này sẽ đề cập trong tài liệu báo cáo
#   unscale dữ liệu
#   Hàm mất mát (Loss Function) - sai số trung bình bình phương (MSE):
#       J = (1/n)((y_true - y_pred)**2).sum()
#2.Đại số tuyến tính:
#   Phương trình hồi quy tuyến tính
#3.Giải tích
#   Đạo hàm riêng
#       delta_J/delta_w = (2/n)*(x*(y_pred - y_true)).sum()
#       delta_J/delta_b = (2/n)*(y_pred - y_true).sum()
#   Thuật toán Gradient Descent
#       Sử dụng các bước đi ngược hướng đạo hàm để tiến dần về cực tiểu
#       w = w - eta.(delta_J/delta_w)
#       b = b - eta.(delta_J/delta_b)
#       (Cách chứng minh sẽ đề cập trong tài liệu báo cáo)
import matplotlib.pyplot as plt #thư viện mạnh cho việc vẽ biểu đồ
import numpy as np #thư viện mạnh cho việc thự hiện các phép toán
import pandas as pd #thư viện mạnh cho việc thao tác và xử lý dữ liệu
import random #để khởi tạo giá trị ngẫu nhiên
from sklearn.linear_model import LinearRegression #dùng mô hình có sẵn từ sklearn
from sklearn.model_selection import train_test_split #để chia dữ liệu theo tỉ lệ

#----------------------------
#MAIN (phần xử lý chính)
#----------------------------
#read csv file
aqi_data_raw = pd.read_csv("btlgt1/Air Quality Ho Chi Minh City.csv")

#dọn dẹp các hàng có giá trị NaN 
aqi_data_cleaned = aqi_data_raw.dropna(how='any', subset=['TSP', 'PM2.5'])

#tính tổng các hàng chứa giá trị NaN còn sót, nếu kết quả bằng 0 thì dữ liệu đã dọn xong
sum_of_NaN_missed = aqi_data_cleaned.isnull().sum().sum()

#chỉ lấy những dữ liệu có giá trị TSP > 0 vì những ô dữ liệu = 0 là dữ liệu rác (dựa trên ý nghĩa thực tế của TSP và PM2.5)
aqi_data_final = aqi_data_cleaned[aqi_data_cleaned['TSP'] > 0]

#trích xuất 2 cột TSP và PM2.5 làm trục X và y (X và y đang là Dataframe nên chưa thể thực hiện tính toán trực tiếp, cần chuyển đổi thành numpy array ở các bước sau)
X,y = aqi_data_final[['TSP']], aqi_data_final['PM2.5']

#tách dữ liệu thành train và test, cho 75% dữ liệu để train, 25% dữ liệu để test, X_Train, X_Test, y_Train, y_Test cũng đang là DataFrame
X_Train, X_Test, y_Train, y_Test = train_test_split(X, y, random_state = 0)

#Tạo model linear regression trong thư viện sklearn để đối chiếu kết quả thủ công
lr = LinearRegression()
lr.fit(X_Train,y_Train)
print("đang huấn luyện...")

#-----------------------------
#Thực hiện thủ công quá trình dùng Gradient Descent để tối ưu hàm mất mát
#-----------------------------
#Viết phương trình hồi quy tuyến tính đơn biến   
w = np.random.randn() #khởi tạo ngẫu nhiên w
b = np.random.randn() # khởi tạo ngẫu nhiên b
n = len(X_Train) #gán n là số mẫu của X_Train
eta = 0.001 #thử eta bằng số này
#Viết hàm thiết lập thang màu đánh giá mức độ bụi mịn
def get_color(value):
    if 0 <= value <= 12.0:
        return '#00e400' # Xanh lá
    elif 12.0 < value <= 35.4:
        return '#ffff00' # Vàng
    elif 35.4 < value <= 55.4:
        return '#ff7e00' # Cam
    elif 55.4 < value <= 150.4:
        return '#ff0000' # Đỏ
    elif 150.4 < value <= 250.4:
        return '#8f3f97' # Tím
    else:
        return '#7e0023'# Nâu sẫm
#tạo danh sách màu tương ứng với từng giá trị pm2.5
color_list = []
for value in y_Train:
    color = get_color(value)
    color_list.append(color)
#-----------------------------
#Viết hàm tính giá trị hàm lỗi
#-----------------------------
def loss_function(y_true,y_pred):#y_true trong bài toán này là y_train, y_pred là y của đường thẳng hồi quy
    return (1/n) * ((y_true - y_pred)**2).sum()
#--------------------------------
#Viết thuật toán gradient descent
#--------------------------------
def gradient_descent(loss_function, eta, w, b): #hàm này truyền vào cả hàm lỗi để liên tục cập nhật w và b dựa trên lý thuyết toán của gradient descent
    for i in range(10000):#cho lặp n lần
        y_line = w * X_Train_np + b #viết phương trình hồi quy tuyến tính trước tiên với w và b đã được khởi tạo ngẫu nhiên, sau đó cập nhật y line
        gradient_loss_w = (2/n) * (X_Train_np * (y_line - y_Train_np)).sum() #đây là biểu thức đạo hàm riêng của hàm loss theo w dùng xtrainscaled để tránh trường hợp bùng nổ gradient
        gradient_loss_b = (2/n) * (y_line - y_Train_np).sum() #tương tự câu trên
        w -= eta * gradient_loss_w #liên tục cập nhật w theo công thức gradient descent
        b -= eta * gradient_loss_b #liên tục cập nhật b theo công thức gradient descent
        error_value_list.append(loss_function(y_Train_np,y_line)) #lưu từng giá trị của hàm loss vào list để tiện vẽ đồ thị hàm loss
        w_list.append(w) #lưu từng giá trị của tham số w
        b_list.append(b) #lưu từng giá trịh của tham số b
    return w,b #sau khi hết 1000 lần lặp thì trả về giá trị w và b cuối cùng (giá trị tiệm cận nhất đến giá trị chính xác)

mu = X_Train.values.mean() #chuyển đổi X_Train về numpy array 1 chiều và thực hiện tính toán giá trị trung bình và gán nó vào mu (bây giờ đang mang kiểu số thực)

sigma = X_Train.values.std() #chuyển đổi X_Train về numpy array 1 chiều và thực hiện tính toán độ lệch chuẩn và gán nó vào sigma (bây giờ đang mang kiểu số thực)

#Scale data bằng phương pháp z score
X_Train_scaled = (X_Train - mu)/sigma #dựa theo công thức đã đề cập, X_Train_scaled vẫn đang là DataFrame do nó tính trên X_Train dataframe

X_Train_np = X_Train_scaled.values.flatten()#chuyển X_Train_scaled sang numpy array
y_Train_np = y_Train.values#chuyển y_Train sang numpy array

#khởi tạo các list để lưu trữ giá trị của hàm lỗi, w và b
error_value_list = []
w_list = []
b_list = []

#gọi hàm gradient descent, tiến hành tối ưu tham số đồng thời lưu w và b mà hàm gradient descent tối ưu vào 2 biến w_opti và b_opti
w_opti, b_opti = gradient_descent(
    loss_function=loss_function,
    eta=eta,
    w=w,
    b=b)

#lúc này w_opti và b_opti là 2 tham số đã được tối ưu nhưng vẫn đang còn ở thanh đo z_score, cần chuyển nó lại (unscale) thang đo gốc
w_unscaled = w_opti/sigma #công thức và cách chứng minh sẽ đề cập trong báo cáo
b_unscaled = b_opti - (w_opti * mu)/sigma

# w_unscaled_list.append(w_list/sigma)
# b_unscaled_list.append(b_list - (w_list * mu)/sigma)
#Viết phương trình hồi quy tuyến tính với w và b đã được tối ưu theo thang đo gốc (microgam/m^30)
y_unscaled = w_unscaled * X_Train + b_unscaled
y_true = lr.coef_ * X_Train + lr.intercept_
print("huấn luyện xong")
print("w và b sau khi tối ưu theo thang scale: ", w_opti,b_opti)
print("w và b sau khi tối ưu theo thang đo gốc (microgam/m^3): ", w_unscaled,b_unscaled)
print("w và b theo thang đo gốc được tối ưu bởi thư viện sklearn: " ,lr.coef_, lr.intercept_)
#----------------------
#BIỂU DIỄN LÊN ĐỒ THỊ
#----------------------
#khởi tạo khung biểu đồ chứa 2 biểu đồ con, 1 chứa biểu đồ đường hồi quy, 1 chứa biểu đồ hàm lỗi
fig, axes = plt.subplots(1,2,figsize=(15,7))

axes[0].set_title("Biểu đồ đường hồi quy tuyến tính")
axes[0].plot(X_Train, y_unscaled, 'red')#vẽ đường hồi quy
axes[0].set_xlabel("TSP (microgam/m^3)")
axes[0].set_ylabel("PM2.5 (microgam/m^3)")
axes[0].grid(True)
axes[0].scatter(
    X_Train,
    y_Train,
    c = color_list,
    alpha = 0.8,
    s=10)#vẽ datapoint

axes[1].set_title("Biểu đồ mô tả quá trình hàm mất mát hội tụ")
axes[1].plot(error_value_list, color='green')#vẽ đồ thị hàm lỗi
axes[1].set_xlabel("epochs")
axes[1].set_ylabel("(microgam/m^3)^2 or MSE")
axes[1].grid(True)

plt.show()