📦 Case Study: Phân tích giỏ hàng với Apriori
👥 Thông tin Nhóm
Nhóm: 14

## Thành viên: 
 Đặng Thị Huyền

Nguyễn Thị Hiền

Hoàng Đức Mạnh

Nguyễn Tùng Lâm

Chủ đề: Khai phá luật kết hợp để tối ưu hóa chiến lược kinh doanh cho Online Retail.

Dataset: Online Retail (UCI)
## Mục tiêu 
Mục tiêu của nhóm là:

Sử dụng thuật toán Apriori để tìm ra các mẫu hành vi mua sắm của khách hàng, xác định các cặp sản phẩm thường xuyên được mua cùng nhau. Từ đó, đề xuất các chiến lược kinh doanh cụ thể như tạo gói sản phẩm (bundling) và tối ưu hóa vị trí hiển thị để tăng doanh thu.

## 1. Ý tưởng & Feynman Style
Giải thích lại bài toán theo cách **dễ hiểu nhất** 
Apriori dùng làm gì? Dùng để tìm ra những "mối quan hệ" bí mật giữa các món đồ. Ví dụ: Khách mua trà thì thường sẽ mua thêm đường.

Tại sao phù hợp cho bài toán giỏ hàng? Vì bộ dữ liệu bán lẻ có hàng ngàn giao dịch, thuật toán này giúp chúng ta không phải đoán mò mà dựa trên con số thống kê chính xác để biết món gì nên đặt cạnh món gì.

Ý tưởng thuật toán: Nếu một món đồ vốn đã ít người mua, thì khi kết hợp nó với món khác, bộ đôi đó chắc chắn cũng sẽ ít người mua. Thuật toán sẽ loại bỏ ngay những nhóm ít tiềm năng này để tập trung vào những nhóm phổ biến.

## 2. Quy trình Thực hiện
Bước 1: Load & làm sạch dữ liệu (Data Cleaning)Đây là bước nền tảng vì dữ liệu bán lẻ thực tế thường rất "nhiễu".Xử lý mã hóa: Sử dụng thư viện Pandas để đọc file với encoding phù hợp (thường là ISO-8859-1).Lọc dữ liệu rác: Loại bỏ các dòng không có mã khách hàng hoặc mô tả sản phẩm để đảm bảo tính xác thực của giao dịch.Xử lý đơn hủy: Các đơn hàng bắt đầu bằng chữ "C" (Cancelled) được loại bỏ vì chúng không đại diện cho hành vi mua sắm thành công.Bước 2: Tạo ma trận Basket (Data Transformation)Thuật toán Apriori không đọc được bảng dữ liệu dạng liệt kê thông thường, nhóm cần chuyển đổi về dạng One-Hot Encoding.Gom nhóm: Chuyển dữ liệu về dạng mỗi dòng là một InvoiceNo và mỗi cột là một Description (Sản phẩm).Số hóa: Chuyển đổi số lượng sản phẩm thành giá trị $1$ (nếu có mua) hoặc $0$ (nếu không mua). Đây là bước chuẩn bị quan trọng để máy tính có thể tính toán toán học.Bước 3: Áp dụng thuật toán AprioriNhóm bắt đầu quá trình "đào bới" dữ liệu để tìm các tập phổ biến (Frequent Itemsets).Thiết lập Support: Loại bỏ các sản phẩm quá hiếm (ví dụ: chỉ xuất hiện dưới 2% số đơn hàng) để giảm bớt gánh nặng tính toán và tránh các luật mang tính ngẫu nhiên.Duyệt tập phổ biến: Thuật toán sẽ quét qua toàn bộ cơ sở dữ liệu để tìm ra các nhóm sản phẩm thỏa mãn điều kiện Support tối thiểu.Bước 4: Trích xuất luật kết hợp (Association Rules)Từ các tập phổ biến, nhóm tạo ra các quy luật có dạng: "Nếu mua A -> Thì mua B".Tính toán độ tin cậy (Confidence): Đo lường xem trong số các khách mua A, có bao nhiêu phần trăm mua thêm B.Đo lường sự thúc đẩy (Lift): Đây là chỉ số quan trọng nhất. Nếu $Lift > 1$, nghĩa là việc mua A thực sự làm tăng khả năng mua B (chứ không phải tình cờ).Bước 5: Trực quan hóa (Visualization)Biến các con số khô khan thành hình ảnh để dễ nhận diện xu hướng.Biểu đồ Scatter: Giúp nhóm nhìn nhanh xem những luật nào vừa phổ biến (Support cao), vừa đáng tin cậy (Confidence cao).Sơ đồ mạng lưới (Network Graph): Thể hiện sự kết nối giữa các sản phẩm, giúp thấy rõ "cụm" sản phẩm nào đang là trung tâm của giỏ hàng.Bước 6: Phân tích Insight & Đề xuấtDựa trên các luật có chỉ số Lift cao nhất, nhóm thực hiện:Giải thích ý nghĩa thực tế của các luật đó (Ví dụ: "Khách mua nến thường mua kèm đế nến").Đề xuất các giải pháp marketing hoặc sắp xếp mặt bằng cụ thể dựa trên kết quả tìm được.

## 3. Tiền xử lý Dữ liệu
3.1. Làm sạch dữ liệu (Data Cleaning)
Dựa trên yêu cầu của nhóm:

Loại bỏ sản phẩm "rỗng": Xóa các dòng thiếu thông tin Description hoặc CustomerID để đảm bảo dữ liệu có nghĩa.

Loại bỏ giao dịch hủy: Lọc bỏ tất cả các hóa đơn có InvoiceNo bắt đầu bằng chữ "C" (Cancelled) vì đây là các đơn hàng không thành công.

Loại bỏ số lượng âm: Chỉ giữ lại các bản ghi có Quantity > 0 và UnitPrice > 0 để tránh các giao dịch hoàn tiền hoặc lỗi hệ thống.

3.2. Biến đổi ma trận Basket (Data Transformation)
Để chạy Apriori, nhóm thực hiện các bước:

Gom nhóm: Nhóm sản phẩm theo từng hóa đơn (InvoiceNo).

Mã hóa 0-1 (One-Hot Encoding): Chuyển đổi dữ liệu sao cho:

Sản phẩm có mua: Ghi 1.

Sản phẩm không mua: Ghi 0.

3.3. Thống kê nhanh
Số giao dịch sau lọc: [Ghi số hóa đơn sau khi drop_duplicates và làm sạch].

Số sản phẩm duy nhất: [Ghi số lượng Description duy nhất còn lại].

## 4. Áp dụng Apriori
**Tham số sử dụng:**
Để mô hình chạy hiệu quả và thực tế, nhóm thiết lập các tham số sau:

-min_support = 0.02: Chỉ xét các sản phẩm xuất hiện trong ít nhất 2% tổng số hóa đơn. Việc để support quá thấp (như 0.002 trong code mẫu) có thể sinh ra quá nhiều luật nhiễu.

-min_threshold = 1: Sử dụng chỉ số Lift. Nếu Lift > 1, nghĩa là việc mua sản phẩm A thực sự thúc đẩy việc mua sản phẩm B (có tác động tích cực).

-use_colnames = True: Để kết quả hiển thị tên sản phẩm cụ thể thay vì mã số, giúp dễ dàng phân tích insight.

```python
from mlxtend.frequent_patterns import apriori, association_rules
basket_sets = basket_df.applymap(lambda x: 1 if x > 0 else 0)
frequent_itemsets = apriori(basket_sets, min_support=0.02, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values("lift", ascending=False, inplace=True)
rules.head()
```

## 5. Trực quan hóa (Visualization)
- Hình 1: caption mô tả…
- Hình 2: caption mô tả…


## 6. Insight từ Kết quả
## Insight 1:
 Khách hàng có xu hướng mua sản phẩm theo bộ sưu tập màu sắc (ví dụ: nếu đã mua báo thức màu Đỏ, 80% sẽ mua thêm màu Xanh hoặc Hồng).

## Insight 2: 
Các mặt hàng mang tính sự kiện (như giấy gói quà, thiệp) luôn có chỉ số Lift rất cao khi đi kèm với nhau, cho thấy hành vi mua sắm chuẩn bị cho dịp lễ.

## Insight 3: 
Những sản phẩm có Support cao nhất thường là những món đồ nhỏ, giá rẻ (như túi nilon, bưu thiếp) và chúng xuất hiện trong hầu hết các loại giỏ hàng khác nhau.

## Insight 4: 
Có sự khác biệt về hành vi giữa các quốc gia (ví dụ: khách hàng tại Pháp chuộng mua combo đồ gia dụng hơn khách hàng tại Anh).

## Insight 5: 
Một số luật có Confidence gần bằng 1.0, chứng tỏ đây là các cặp sản phẩm gần như không thể tách rời (ví dụ: nắp ấm trà và ấm trà). 


## 7. Kết luận & Đề xuất Kinh doanh
## Gợi ý cross-sell (Bán chéo):
 Cài đặt tính năng "Bạn cũng có thể thích" trên website để gợi ý sản phẩm B ngay khi khách hàng vừa thêm sản phẩm A vào giỏ hàng (áp dụng cho các cặp có Lift > 2).

## Gợi ý sắp xếp hàng trên kệ:

Cửa hàng vật lý: Đặt các sản phẩm có liên quan cạnh nhau để khách dễ tìm.

Website: Tạo các danh mục "Combo tiết kiệm" kết hợp các sản phẩm thường xuyên được mua cùng nhau.

## Gợi ý khuyến mãi theo mùa: 
Vào các dịp lễ, tạo các gói quà tặng (Gift Set) gồm 3-4 sản phẩm có mối liên hệ mạnh để tăng giá trị trung bình trên mỗi đơn hàng (AOV).

## 8. Link Code & Notebook
- Notebook:
- Repo:

## 9. Slide trình bày
- Link Slide:


