# Bài toán cho dữ liệu bệnh nhân trước, xây dựng model dự đoán khả năng bị bệnh tim của một số bệnh nhân
# Quy ước : nhẹ - 1, thấp - 2, trung bình - 3, cao - 4, nặng - 5, ít - 6, nhiều - 7
# Quy ước : Bị bênh tim - 1, Không bị bệnh - 0
# Dữ liệu ban đầu về các tiêu chí
'''
Các bước xử lí bài toán:
B1: Thu thập dữ liệu.
B2: Xử lí dữ liệu.
B3: Xây dựng model.
B4: Dự đoán kết quả.
B5: Đánh giá kết quả.
'''

from sklearn import tree
my_tree = tree.DecisionTreeClassifier()
dactrung = [
			[1, 3, 3, 7],
			[5, 2, 4, 6],
			[1, 2, 4, 6],
			[5, 4, 4, 3],
			[1, 4, 4, 7],
			[3, 2, 3, 7],
			[3, 3, 3, 6],
			[5, 2, 2, 7]
]
# Kết quả bị bệnh (nhãn) tương ứng với dữ liệu trên
nhan = [0, 1, 1, 0, 0, 0, 0, 1]

result = my_tree.fit(dactrung, nhan)

kq1, kq2 = result.predict([[1, 4, 3, 6], [1, 4, 3, 7]])
print(kq1, kq2)