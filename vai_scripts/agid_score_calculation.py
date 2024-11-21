import pandas as pd
import numpy as np
####################################################################################################################################################### change csv file name accordingly
df2=pd.read_csv(r'/path/to/your/file/indexing/Twitter_DALLE3.csv')
L = df2.min(numeric_only=True)
mu = df2.mean(numeric_only=True)
delta = [1, 1, 1, 1, 1, 1, 1]
def calculate_adi(row, L, mu, delta):
    N = 1
    indices = row[1:].values
    adi_score = 100 / (N ** 2) * sum(delta[i] * (indices[i] - L[i]) / (1 - mu[i]) for i in range(len(indices)))
    return adi_score
df2['ADI_Score'] = df2.apply(calculate_adi, axis=1, args=(L, mu, delta))
def scale_adi_score(adi_score):
    min_adi = df2['ADI_Score'].min()
    max_adi = df2['ADI_Score'].max()
    scaled_adi_score = (100 * (adi_score - min_adi) / (max_adi - min_adi))
    return scaled_adi_score
df2['Scaled_ADI_Score'] = df2['ADI_Score'].apply(scale_adi_score)
############################################################################################# change csv file name accordingly
df2[['Filename', 'Scaled_ADI_Score']].to_csv('/path/to/your/file/indexing/Twitter_DALLE3_score.csv',index=False)
#################################################################### change model name accordingly
print('Average score for Twitter_DALLE3 is', df2['Scaled_ADI_Score'].mean())
