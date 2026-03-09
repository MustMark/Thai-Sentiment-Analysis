import pandas as pd
import json
import re

print("กำลังโหลดข้อมูล train_sentiment.json...")
with open('train_sentiment.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# แปลงเป็น DataFrame เพื่อให้จัดการง่ายขึ้น
df = pd.DataFrame(data)

print(f"จำนวนข้อมูลก่อนทำความสะอาด: {len(df)} บรรทัด")

# ==========================================
# กฎข้อที่ 1: แก้ไขข่าว PR / แคมเปญ (Neutral -> Positive)
# ==========================================
positive_keywords = ['คิกออฟ', 'โครงการ', 'แคมเปญ', 'แจกฟรี', 'ฟรี !!!', 'มอบ', 'ความสุข', 'ช่วยเหลือ', 'โปรโมชัน', 'ส่วนลด', 'จิตอาสา', 'ปรับปรุงภูมิทัศน์']

def fix_positive(row):
    text = str(row['text'])
    # ถ้าป้ายเดิมคือ neutral และมีคำที่เป็นแง่บวก/PR
    if row['sentiment'] == 'neutral' and any(kw in text for kw in positive_keywords):
        # ข้อยกเว้น: ถ้าในประโยคมีคำว่าไฟไหม้/ฝุ่น ห้ามแก้เป็นบวกเด็ดขาด
        if not any(bad_kw in text for bad_kw in ['ฝุ่น', 'ไฟไหม้', 'เพลิงไหม้', 'PM 2.5']):
            return 'positive'
    return row['sentiment']

df['sentiment'] = df.apply(fix_positive, axis=1)

# ==========================================
# กฎข้อที่ 2: แก้ไขข่าวภัยพิบัติ / ฝุ่น (Neutral -> Negative)
# ==========================================
negative_keywords = ['PM 2.5', 'ฝุ่น', 'เพลิงไหม้', 'ไฟไหม้', 'อุบัติเหตุ', 'เสียชีวิต', 'ร้องเรียน', 'เดือดร้อน', 'รกร้าง', 'น้ำท่วม', 'แล้งหนัก']

def fix_negative(row):
    text = str(row['text'])
    if row['sentiment'] == 'neutral' and any(kw in text for kw in negative_keywords):
        return 'negative'
    return row['sentiment']

df['sentiment'] = df.apply(fix_negative, axis=1)

# ==========================================
# กฎข้อที่ 3: ลบข้อมูลขยะ (Drop Outliers)
# ==========================================
def is_garbage(text):
    text = str(text)
    # 3.1 ลบข่าวภาษาอังกฤษล้วน (ไม่มีพยัญชนะไทยเลย หรือมีคำว่า Cambodia ที่เป็นข่าวที่เราเจอ)
    if 'Cambodia' in text and 'internet' in text:
        return True
    
    # เช็คว่ามีตัวอักษรภาษาไทยในประโยคหรือไม่ (ถ้าไม่มีเลย = ขยะ)
    if not re.search(r'[ก-๙]', text):
        return True
        
    return False

# กรองเอาเฉพาะบรรทัดที่ไม่ใช่ขยะ (is_garbage == False)
df = df[~df['text'].apply(is_garbage)]

print(f"จำนวนข้อมูลหลังทำความสะอาด: {len(df)} บรรทัด")

# ==========================================
# บันทึกกลับเป็นไฟล์ JSON คลีนๆ
# ==========================================
cleaned_data = {
    "text": df['text'].to_dict(),
    "sentiment": df['sentiment'].to_dict()
}

with open('train_sentiment_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=4)

print("✅ ทำความสะอาดเสร็จสิ้น! บันทึกไฟล์ใหม่ชื่อ 'train_sentiment_cleaned.json' เรียบร้อยแล้ว")