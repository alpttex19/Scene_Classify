import pandas as pd

# 读取CSV文件
df = pd.read_csv('filtered_test.csv')

# 删除第二列（label）为1的所有行
df = df[df['label'] != 1]

# 保存处理后的CSV文件
df.to_csv('test_data.csv', index=False)
