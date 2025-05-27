import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み込み
image = cv2.imread('metal_panel.jpg')

# グレースケール変換
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二値化（白色領域を抽出）
#_, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
_, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_OTSU)

# カーネル（構造要素）の定義：3x3の正方形
kernel = np.ones((3,3), np.uint8)
# 膨張（白の領域が広がる）
eroded = cv2.erode(binary, kernel, iterations=10)
# 収縮（白の領域が縮む）
dilated = cv2.dilate(eroded, kernel, iterations=10)


# ノイズ除去のためのモルフォロジー処理
kernel = np.ones((5, 5), np.uint8)
cleaned_binary = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

#　輪郭の抽出
#contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 対象以外の領域を除外
min_area = 1000  # 最小面積の閾値
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# マスク画像の作成
mask = np.zeros_like(gray)
cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

# 金属部分のみの表示(元画像が大きい為、画像表示を確認できるサイズに縮小)
masked_image = cv2.bitwise_and(image, image, mask=mask)
resized_image = cv2.resize(masked_image, (350, int(masked_image.shape[0] * 350 / masked_image.shape[1])))

# 結果の表示
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()