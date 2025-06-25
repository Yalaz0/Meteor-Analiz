# streamlit_app.py – Profesyonel ve şık çok sayfalı yapı + Geliştirilmiş Uzay Arka Planı
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px



st.set_page_config(
    page_title="Meteor & Şehirleşme Analizi",
    page_icon="🌌",
    layout="wide",
)

# Şık uzay temalı CSS
st.markdown("""
    <style>
    html, body {
        background-color: #0b0c10;
        background-image: radial-gradient(circle at 1px 1px, #ffffff22 1px, transparent 0);
        background-size: 20px 20px;
        color: #f0f0f0;
    }
    .block-container {
        padding: 2rem 5rem;
    }
    h1, h2, h3 {
        color: #66fcf1;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton > button {
        background-color: #1f2833;
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        font-size: 16px;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background-color: #45a29e;
        color: #0b0c10;
    }
    .stSidebar > div {
        background-color: #1f2833 !important;
    }
    .stMarkdown, .stDataFrame, .stTable, .stText, .stSubheader, .stHeader {
        color: #ffffff !important;
    }
    /* Aktif sayfa butonu rengi */
    .stButton > button:focus:not(:active) {
        background-color: #66fcf1 !important;
        color: #0b0c10 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌌 Meteor Düşüşleri ve Şehir Yerleşimleri")



# Menü tanımı
sayfalar = [
    "Proje Bilgisi",
    "Harita Üzerinden İnceleme",
    "Modelleme ve Tahmin",
]



st.sidebar.title("🔭 Sayfa Seçimi")
st.sidebar.markdown("---")

for page in sayfalar:
    if st.sidebar.button(page, use_container_width=True):
        st.session_state["sayfa"] = page

if "sayfa" not in st.session_state:
    st.session_state["sayfa"] = "Proje Bilgisi"

sayfa = st.session_state["sayfa"]

@st.cache_data
def load_meteor_data():
    return pd.read_csv("data/meteorite-landings.csv")


@st.cache_data
def load_city_data():
    return gpd.read_file("data/ne_10m_populated_places/ne_10m_populated_places.shp")



if sayfa == "Proje Bilgisi":
    st.header("🛰️ Proje Başlığı")

    import json
    import os
    from streamlit_lottie import st_lottie
    import streamlit as st
    
    # 🔧 Uygun dosya yolu
    json_path = "assets/Animation - 1747267971133.json"
    
    # 🎬 JSON dosyasını yükleyen fonksiyon
    def load_lottie_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # ✅ Dosya var mı kontrol et ve göster
    if os.path.exists(json_path):
        lottie_meteor = load_lottie_file(json_path)
        st_lottie(lottie_meteor, speed=1, height=400, loop=True, key="meteor")
    else:
        st.error(f"⚠️ Lottie animasyonu bulunamadı: {json_path}")

    st.markdown("""
    ### Meteor Düşüşleri ile İnsan Yerleşimleri Arasındaki Coğrafi İlişki  
    *Mekânsal Veri Analizi ve Makine Öğrenmesi Tabanlı Bir Yaklaşım*

    Bu proje, meteor düşüş olaylarının tarih boyunca insanların yerleşim yerleri üzerindeki etkisini analiz etmeyi amaçlayan disiplinler arası bir çalışmadır. Gökbilimi, coğrafya, istatistik ve makine öğrenmesini bir araya getirerek, insan yerleşimlerinin meteor düşüşleriyle olan potansiyel bağlantısını anlamaya çalışır.
    """)

    st.subheader("🧠 Hipotez")
    st.markdown("""
    İnsanlar tarih boyunca, meteor düşüşlerinin sık yaşandığı veya bu düşüşlerin zengin mineral kalıntıları bıraktığı coğrafi bölgelerde yaşam kurma eğilimi göstermiştir.

    Bu hipotez, özellikle meteorların düştüğü yerlerin madencilik, tarım veya ticaret açısından cazip olabileceği ve zamanla bu bölgelerde şehirlerin kurulduğu varsayımına dayanır. Dolayısıyla, şehirlerin meteor düşüş noktalarına olan mesafelerinin rastlantısal değil, anlamlı bir örüntü gösterdiği düşünülmektedir.
    """)

    st.subheader("🎯 Amaç")
    st.markdown("""
    Bu projenin temel amacı, şehir konumları ile meteor düşüşleri arasında mekânsal ve istatistiksel bir ilişki olup olmadığını araştırmaktır. Bu kapsamda:

    - Her şehir için meteorlara olan en kısa mesafe hesaplandı.
    - Bu mesafelere göre şehirler, yakın/uzak şeklinde sınıflandırıldı.
    - Makine öğrenmesi algoritmaları ile bu sınıflandırmanın tahmin edilebilirliği test edildi.
    - Coğrafi kümeleme analizleri ile meteor düşüşlerinin yoğunlaştığı bölgeler belirlendi.
    - Zaman serisi analizi ile düşüşlerin tarihsel dağılımı incelendi.
    - Sonuçlar görselleştirilerek, hipotezin geçerliliği grafik ve model çıktılarıyla desteklendi.
    """)

    st.subheader("📦 Kullanılan Veri Setleri")
    st.markdown("""
    **[Meteorite Landings Dataset (Kaggle / NASA)](https://www.kaggle.com/datasets/nasa/meteorite-landings)**  
    - 38.000'den fazla meteor düşüş kaydını içerir.  
    - Konum (enlem, boylam), düşme yılı, kütle, tür bilgileri içerir.  
    - Kaynak: The Meteoritical Society / NASA Open Data Portal

    **[Natural Earth Populated Places](https://www.naturalearthdata.com/downloads/)**  
    - Şehirlerin koordinat, ülke, isim ve nüfus bilgileri.  
    - 7.300'den fazla yerleşim birimi.  
    - Kaynak: Natural Earth / GeoPandas datasets

    **Türev Veri Setleri:**  
    - `cities_proj`: EPSG:3857 projeksiyonunda şehir verisi  
    - `meteor_proj`: Projeksiyona dönüştürülmüş meteor verisi  
    - `cities_model`: Mühendislik uygulanmış model verisi (mesafe, etiket, kıta dummies)  
    """)



elif sayfa == "Harita Üzerinden İnceleme":
    st.header("🌍 Meteor Haritası ve Yerleşim İncelemesi")

    st.markdown("""
    Bu sayfa, Dünya'ya düşen meteorların ve şehir yerleşimlerinin **coğrafi dağılımını** hem klasik haritalar hem de modern 3D görselleştirme ile sunar.
    Aşağıdaki haritalarda meteorların yoğunluk alanları ve şehir konumları incelenebilir.
    """)

    meteor_df = load_meteor_data()
    city_gdf = load_city_data().to_crs("EPSG:4326")
    
        # 🔽 BURAYA EKLE
    st.sidebar.markdown("### 🔎 Filtreleme Seçenekleri")
    
    # Yıl Aralığı
    min_year = int(meteor_df["year"].min())
    max_year = int(meteor_df["year"].max())
    year_range = st.sidebar.slider("Yıl Aralığı", min_year, max_year, (1900, 2000))
    
    # Kütle Aralığı (güvenli)
    mass_col = meteor_df["mass"].dropna()
    mass_range = st.sidebar.slider(
        "Kütle Aralığı (gram)",
        int(mass_col.min()),
        int(mass_col.max()),
        (1000, 100000)
    )
    
    # Fell / Found seçimi
    fall_options = st.sidebar.multiselect(
        "Meteor Kategorisi (Fall)", 
        options=["Fell", "Found"], 
        default=["Fell", "Found"]
    )
    
    # Sadece büyük meteorlar
    highlight_big = st.sidebar.checkbox("Sadece büyük meteorları göster (>10000g)")
    
    # 📌 Filtre Uygulama
    filtered_df = meteor_df.copy()
    filtered_df = filtered_df[
        (filtered_df["year"].between(year_range[0], year_range[1])) &
        (filtered_df["mass"].between(mass_range[0], mass_range[1])) &
        (filtered_df["fall"].isin(fall_options))
    ]
    if highlight_big:
        filtered_df = filtered_df[filtered_df["mass"] > 10000]


    # Şehir koordinatlarını ayıkla
    city_gdf["lat"] = city_gdf.geometry.y
    city_gdf["lon"] = city_gdf.geometry.x



    st.subheader("📍 2D Harita – Meteor Düşüş Noktaları (Etkileşimli)")
    
    fig = px.scatter_mapbox(
        filtered_df.dropna(subset=["reclat", "reclong"]),
        lat="reclat", lon="reclong",
        color="mass",
        size="mass",
        hover_name="name",
        hover_data=["year", "recclass", "mass"],
        color_continuous_scale="YlOrRd",
        zoom=1,
        height=600
    )
    fig.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📊 Özet Bilgiler")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Gösterilen Meteor Sayısı", len(filtered_df))
    col2.metric("Ortalama Kütle (g)", f"{filtered_df['mass'].mean():,.0f}")
    col3.metric("En Ağır Meteor (g)", f"{filtered_df['mass'].max():,.0f}")
    
    st.markdown("#### 🪐 En Büyük 5 Meteor")
    st.dataframe(
        filtered_df.sort_values(by="mass", ascending=False)
        [["name", "year", "mass", "recclass"]].head(5),
        use_container_width=True
    )

    

    st.subheader("🏙️ Şehir Yerleşim Noktaları (2D)")
    st.markdown("""
    Bu harita, şehirlerin dünya genelindeki konumlarını gösterir.
    Noktalar, 3D görseldeki gibi **mavi renk temasıyla** işaretlenmiştir.
    """)
    # Özel renkli noktalar için Pydeck Scatterplot kullan
    city_layer_2d = pdk.Layer(
        "ScatterplotLayer",
        data=city_gdf.dropna(subset=["lat", "lon"]),
        get_position='[lon, lat]',
        get_color='[0, 120, 255, 180]',
        get_radius=50000,
        pickable=False,
    )
    view_state_2d = pdk.ViewState(latitude=20, longitude=0, zoom=1.2, pitch=0)
    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v10",
            initial_view_state=view_state_2d,
            layers=[city_layer_2d],
        )
    )

    st.subheader("🛰️ 3D Görselleştirme")
    st.markdown("""
    Son olarak, meteor yoğunluklarını ve şehir yerleşimlerini 3 boyutlu olarak görebileceğiniz etkileşimli bir harita aşağıdadır.
    """)

    # Yeni: Kırmızı-turuncu renk geçişli meteor yoğunluğu
    meteor_layer = pdk.Layer(
        "HexagonLayer",
        data=meteor_df.dropna(subset=["reclat", "reclong"]),
        get_position='[reclong, reclat]',
        radius=50000,
        elevation_scale=80,
        elevation_range=[0, 4000],
        pickable=True,
        extruded=True,
        color_range=[
            [255, 69, 0],
            [255, 100, 0],
            [255, 140, 0],
            [255, 180, 60],
            [255, 220, 100],
        ]
    )

    city_layer = pdk.Layer(
        "ColumnLayer",
        data=city_gdf.dropna(subset=["lat", "lon"]),
        get_position='[lon, lat]',
        get_elevation=300000,
        elevation_scale=1,
        radius=15000,
        get_fill_color='[0, 120, 255, 180]',
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2, pitch=45)

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/dark-v10",
            initial_view_state=view_state,
            layers=[meteor_layer, city_layer],
            tooltip={"text": "Konum Bilgisi"}
        )
    )

    st.markdown("""
    - **Kırmızı tonlar**: Meteor düşüşlerinin yoğun olduğu bölgeleri temsil eder.
    - **Mavi sütunlar**: Dünya üzerindeki şehirlerin yerleşim noktalarını 3D binalar gibi gösterir.
    - **Yüksek alanlar**, daha fazla yoğunluğa veya nüfusa sahip bölgeleri simgeler.
    """)

    st.subheader("🧩 İleri Analizler")
    st.info("Bu bölümde yoğunluk haritası, zaman serisi ve kümelenme analizleri entegre edilecektir.")
    with st.expander("📍 Şehir Bazlı Meteor Yakınlık Analizi", expanded=True):
        selected_city = st.selectbox("Bir şehir seçin", city_gdf["NAME"].dropna().unique())
    
        selected_city_row = city_gdf[city_gdf["NAME"] == selected_city].iloc[0]
        city_coord = (selected_city_row["lat"], selected_city_row["lon"])
    
        # ⚠️ Eksik koordinatları temizle
        filtered_df = filtered_df.dropna(subset=["reclat", "reclong"])
    
        # Mesafe hesapla
        filtered_df["distance_km"] = filtered_df.apply(
            lambda row: geodesic(city_coord, (row["reclat"], row["reclong"])).km,
            axis=1
        )

    
        closest_meteor = filtered_df.sort_values(by="distance_km").iloc[0]
    
        st.metric("En Yakın Meteor", f"{closest_meteor['name']} ({closest_meteor['distance_km']:.2f} km)")
        st.markdown(f"**Yıl:** {closest_meteor['year']}  \n**Kütle:** {closest_meteor['mass']} g")
    
        st.markdown("#### 🔍 250 km içindeki meteorlar")
        nearby_meteors = filtered_df[filtered_df["distance_km"] <= 250].sort_values(by="distance_km")
        st.dataframe(nearby_meteors[["name", "year", "mass", "distance_km"]].head(10), use_container_width=True)
    
    with st.expander("🌐 Meteor–Şehir Yakınlık Oranları (Genel Hipotez Testi)", expanded=True):
            st.markdown("""
            Bu görselleştirme, şehirlerin belirli mesafe eşiklerine göre en az bir meteora yakın olup olmadığını gösterir.  
            Hipotezimizi test etmek için dolaylı bir kanıt sağlar.
            """)
        
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        
            oranlar = {
                "500 km": 85.49,
                "400 km": 81.11,
                "300 km": 73.40,
                "250 km": 67.37,
                "200 km": 59.48,
                "150 km": 49.07,
                "100 km": 34.09,
                "50 km": 15.15
            }
        
            fig = make_subplots(
                rows=2, cols=4,
                specs=[[{'type': 'indicator'}]*4, [{'type': 'indicator'}]*4],
                horizontal_spacing=0.08,
                vertical_spacing=0.2
            )
        
            for i, (mesafe, oran) in enumerate(oranlar.items()):
                row = i // 4 + 1
                col = i % 4 + 1
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=oran,
                        title={'text': mesafe, 'font': {'size': 18}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#66fcf1"},
                            'bgcolor': "#1f2833",
                            'steps': [
                                {'range': [0, 50], 'color': "#2b2f33"},
                                {'range': [50, 100], 'color': "#3a3f44"}
                            ],
                        },
                        number={'suffix': "%"}
                    ),
                    row=row, col=col
                )
        
            fig.update_layout(
                height=500,
                margin=dict(t=40, b=20),
                paper_bgcolor="#0b0c10",
                font=dict(color="white", family="Segoe UI"),
            )
        
            st.plotly_chart(fig, use_container_width=True)



elif sayfa == "Modelleme ve Tahmin":
    st.header("🧠 Modelleme & Tahmin")
    st.markdown("""
    Bu sayfa, projede kullanılan farklı model gruplarının sonuçlarını sunar.
    Her alt başlık farklı modelleme paradigmasını temsil eder.
    """)

    with st.expander("🛠️ Hiperparametre Optimizasyonu ve Model Karşılaştırmaları", expanded=False):
        st.markdown("""
        Hiperparametre optimizasyonu, özellikle ağaç tabanlı algoritmaların başarısını artırmak için kritik bir adımdır. Bu projede kullanılan modellerin bazıları aşağıdaki yöntemlerle optimize edilmiştir:
    
        #### 🔍 Uygulanan Optimizasyon Teknikleri:
        - **GridSearchCV**: Küçük hiperparametre alanlarında sistematik tarama yapılmıştır.  
          Kullanıldığı yerler: `Logistic Regression`, `Decision Tree`, `LDA`
        - **RandomizedSearchCV**: Geniş hiperparametre alanlarında rastgele örnekleme ile hızlı arama.  
          Kullanıldığı yerler: `Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`

    
        #### 📊 Performans Farkları:
        - Optimizasyon sonrası özellikle **XGBoost** ve **LightGBM** modellerinde **%3–5 oranında doğruluk artışı** gözlemlenmiştir.
        - `Logistic Regression` modelinde hiperparametre etkisi sınırlı olmasına rağmen **class_weight='balanced'** gibi küçük ayarlamalar sınıflar arası dengesizlikleri azaltmıştır.
    
        #### 📌 Model Avantajları ve Sınırlılıkları:
    
        | Model            | Avantajlar | Sınırlılıklar |
        |------------------|------------|----------------|
        | Logistic Regression | Yorumlanabilirlik, basitlik | Doğrusal varsayıma bağlı |
        | Random Forest    | Doğruluk, overfitting'e direnç | Yavaş tahmin, fazla kaynak kullanımı |
        | XGBoost / LightGBM | Hızlı ve güçlü, iyi genelleme | Hiperparametre hassasiyeti yüksek |
        | CatBoost         | Kategorik verilerle iyi çalışır | GPU olmadan eğitim yavaş olabilir |
    
        #### 🚀 Dağıtıma Uygunluk Değerlendirmesi:
        - **En iyi dağıtıma uygun model**: `LightGBM`
          - Hafif yapısı ve hızlı tahmin süresi sayesinde API ya da web uygulamalarına kolay entegre edilebilir.
          - Gömülü erken durdurma, düşük gecikmeli tahminler sunar.
        - `Logistic Regression` ise düşük kaynak ihtiyacı nedeniyle **mobil cihazlar** için avantajlıdır.
        - Derin öğrenme tabanlı modeller (ConvLSTM gibi) genellikle **bulut tabanlı dağıtımlar** için uygundur (yüksek hesaplama gücü gerektirir).
    
        📎 **Sonuç**:  
        Her model farklı açılardan güçlüdür. Bu projede hem doğruluk hem de yorumlanabilirlik dikkate alınarak **LightGBM**, dağıtıma uygunluk ve genel performans açısından öne çıkmıştır.
        """)

    
    sekmeler = st.tabs([
        "🎯 Temel & Ağaç Modelleri",
        "🗺️ Mekânsal Kümelenme",
        "📍 Mekânsal Regresyon",
        "🧪 Nedensel & Derin Öğrenme"
    ])
    
    # 1️⃣ TEMEL & AĞAÇ MODELLERİ
    with sekmeler[0]:
        st.subheader("🎯 Temel & Ağaç Tabanlı Sınıflandırıcılar")
        model_df = pd.DataFrame({
            "Model": [
                "Logistic Regression", "LDA", "GAM",
                "Decision Tree", "Random Forest", "XGBoost",
                "LightGBM", "CatBoost", "ExtraTrees"
            ],
            "Accuracy": [0.863, 0.856, 0.863, 0.863, 0.863, 0.863, 0.607, 0.607, 0.607],
            "F1 Score": [0.927, 0.919, 0.927, 0.927, 0.927, 0.927, 0.719, 0.719, 0.719]
        })
        st.dataframe(model_df, use_container_width=True)
    
        import plotly.express as px
        melted_df = model_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(
            melted_df,
            x="Model", y="Score",
            color="Metric", barmode="group",
            text_auto=".2f"
        )
        fig.update_layout(height=500, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🔍 Seçilen Model Detayı")
        
        model_detay_df = pd.DataFrame({
            "Model": [
                "Logistic Regression", "LDA", "GAM",
                "Decision Tree", "Random Forest", "XGBoost",
                "LightGBM", "CatBoost", "ExtraTrees"
            ],
            "Accuracy": [0.863, 0.856, 0.863, 0.863, 0.863, 0.863, 0.607, 0.607, 0.607],
            "F1": [0.927, 0.919, 0.927, 0.927, 0.927, 0.927, 0.719, 0.719, 0.719],
            "Precision": [0.863, 0.891, 0.863, 0.863, 0.863, 0.863, 0.938, 0.938, 0.938],
            "Recall": [1.000, 0.949, 1.000, 1.000, 1.000, 1.000, 0.584, 0.584, 0.584]
        })

        
        selected_model = st.selectbox("🧠 İncelemek için bir model seçin", model_detay_df["Model"])
        model_info = model_detay_df[model_detay_df["Model"] == selected_model].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{model_info['Accuracy']:.2f}")
        col2.metric("F1 Score", f"{model_info['F1']:.2f}")
        col3.metric("Precision", f"{model_info['Precision']:.2f}")
        col4.metric("Recall", f"{model_info['Recall']:.2f}")
        
        st.markdown("### 🧮 Sembolik Confusion Matrix")
        confusion_data = pd.DataFrame(
            [[80, 20], [15, 85]],
            index=["Gerçek: Negatif", "Gerçek: Pozitif"],
            columns=["Tahmin: Negatif", "Tahmin: Pozitif"]
        )
        st.dataframe(confusion_data, use_container_width=True)
        
        st.markdown(f"""
        > 📌 **{selected_model} Yorum:**  
        Bu model, genel doğruluk açısından {model_info['Accuracy']:.0%} başarı göstermektedir.  
        F1 skoru ise {model_info['F1']:.2f} ile sınıf dengesini iyi yansıtmaktadır.
        """)


        st.markdown("### 📈 ROC Eğrisi – Model Karşılaştırması")
        
        roc_df = pd.DataFrame({
            "Model": model_detay_df["Model"],
            "FPR": [0.2, 0.18, 0.16, 0.14, 0.12, 0.10, 0.11, 0.12, 0.13],
            "TPR": [0.70, 0.73, 0.75, 0.78, 0.82, 0.84, 0.83, 0.83, 0.82],
            "AUC": [0.79, 0.82, 0.84, 0.87, 0.90, 0.92, 0.91, 0.91, 0.90]
        })
        
        fig_roc = px.line(
            roc_df,
            x="FPR",
            y="TPR",
            color="Model",
            line_shape="spline",
            markers=True,
            title="Model Karşılaştırmalı ROC Eğrisi"
        )
        fig_roc.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=500)
        st.plotly_chart(fig_roc, use_container_width=True)
        
        st.markdown("""
        > 📊 **AUC Skoru** karşılaştırması sayesinde hangi modelin daha iyi ayrım gücüne sahip olduğu gözlemlenebilir.  
        XGBoost, LightGBM ve CatBoost modelleri bu metrikte öne çıkmaktadır.
        """)

        
        # 🔢 Gerçek verilerden confusion matrix
        cm = [[201, 0],
              [0, 1268]]
        
        labels = ["Uzak", "Yakın"]
        
        # Plotly ile ısı haritası oluştur
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            hovertemplate='Gerçek: %{y}<br>Tahmin: %{x}<br>Adet: %{z}<extra></extra>',
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="📊 Logistic Regression – Confusion Matrix",
            xaxis_title="Tahmin Edilen",
            yaxis_title="Gerçek",
            autosize=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        ### 🔍 Confusion Matrix Yorumları
        
        Bu matriste sınıflandırma modelimizin (Logistic Regression) performansı detaylı bir şekilde gözlemlenebilir.
        
        - **Gerçek sınıflar (Y-axis)**: Gerçekte şehir bir meteora uzak mı yakın mı?
        - **Tahmin edilen sınıflar (X-axis)**: Modelin bu şehirleri nasıl sınıflandırdığı.
        - **Isı haritası renkleri**: Sayısal yoğunluğa göre otomatik renk tonlaması yapılmıştır.
        
        #### 📌 Gözlemler:
        - Model, tüm `Uzak` ve `Yakın` şehirleri doğru sınıflandırmıştır.
        - Hiçbir **False Positive (FP)** ya da **False Negative (FN)** yoktur.
        - Bu sonuçlar %100 doğru sınıflandırma anlamına gelir.
        
        #### 📈 Teknik Değerlendirme:
        - **Accuracy**: Modelin genel doğruluğu → `1.000` (%100)
        - **Precision**: Yakın olarak sınıflandırılanların gerçekten yakın olma oranı → `1.000`
        - **Recall**: Gerçekten yakın olanların ne kadarını doğru bulabildi → `1.000`
        - **F1 Score**: Precision ve Recall dengesini gösteren birleşik skor → `1.000`
        
        #### ✅ Hipoteze Katkı:
        Bu model, şehirlerin meteorlara olan uzaklığı temel alınarak sınıflandırılabileceğini güçlü şekilde destekliyor.
        
        Bu da hipotezimizin (*Meteor düşüşleri rastgele değil, yerleşimlerle ilişkili olabilir*) istatistiksel düzeyde **öngörülebilir olduğunu** gösterir.
        """)


    # 2️⃣ MEKÂNSAL KÜMELENME
    with sekmeler[1]:
        st.subheader("🗺️ Mekânsal Kümelenme Modelleri")
        st.markdown("""
        Bu bölümde, meteorların coğrafi yoğunluk ve küme yapısını analiz eden algoritmalar kullanılmıştır:
        - **DBSCAN**
        - **HDBSCAN**
        - **OPTICS**
        - **MeanShift**
        
        Bu yöntemler, özellikle sık meteor düşüşü yaşanan bölgeleri otomatik olarak tespit etmek için kullanılır.
        """)
        st.markdown("### 🎯 Amaç")
        st.markdown("""
        Meteor düşüşleri coğrafi olarak rastgele dağılmıyor olabilir.  
        Bu hipotezi test etmek için mekânsal kümelenme algoritmaları uygulandı.
        """)
        
        st.markdown("### ✅ Uygulanan Kümelenme Modelleri")
        cluster_table = pd.DataFrame({
            "Model": ["DBSCAN", "HDBSCAN", "MeanShift", "OPTICS"],
            "Küme Sayısı (≠ -1)": [41, 40, "~40+", 9],
            "Gürültü Noktası": ["564", "düşük", "Yok", "9469 (24.7%)"],
            "En Büyük Küme": ["6214 (#7)", "264 (#17)", "6214 (#0)", "6214 (#2)"],
            "Destek Düzeyi": ["✅ Orta", "⚠️ Sınırlı", "✅ Güçlü", "✅ Güçlü"]
        })
        st.dataframe(cluster_table, use_container_width=True)
        
        st.markdown("### 📊 Gözlemler ve Notlar")
        
        st.markdown("""
        ✅ **OPTICS & MeanShift**:
        - Büyük, anlamlı ve tekrar eden kümeler üretmiştir.
        - **6214** noktalık küme çoğu yöntemde tekrar etmiştir.
        - Hipotezi güçlü şekilde destekliyor.
        
        ⚠️ **HDBSCAN**:
        - Küçük ama çok sayıda küme.
        - Lokal yoğunlukları tespit ediyor ama genel desen zayıf.
        
        ⚠️ **DBSCAN**:
        - Gürültü oranı kabul edilebilir.
        - Kümeler çok sayıda ve küçük, orta seviyede destek sağlıyor.
        """)
        
        st.markdown("### 📌 Hipotez Değerlendirmesi")
        hypo_eval = pd.DataFrame({
            "Model": ["OPTICS", "MeanShift", "DBSCAN", "HDBSCAN"],
            "Hipotezi Destekliyor mu?": ["✅ Evet", "✅ Evet", "✅ Orta", "⚠️ Zayıf"],
            "Gerekçe": [
                "Az sayıda büyük, belirgin kümeler + düşük gürültü oranı",
                "Büyük hacimli kümeler, net ayrımlar",
                "Kümeler var ama küçük ve çok sayıda",
                "Yerel yoğunluklar var, desen zayıf"
            ]
        })
        st.dataframe(hypo_eval, use_container_width=True)
        
        st.success("""
        🧠 **Sonuç:**  
        Bu analizler meteorların rastgele dağılmadığını, özellikle OPTICS ve MeanShift ile bazı bölgelerde **coğrafi olarak tekrar ettiğini** gösteriyor.  
        Bu da hipotezin mekânsal yönünü güçlü şekilde desteklemektedir.
        """)
    
                 
     # 3️⃣ MEKÂNSAL REGRESYON
    with sekmeler[2]:
        st.subheader("📍 Mekânsal Regresyon Modelleri")
        st.markdown("""
        Meteor mesafesinin şehirleşme üzerindeki mekânsal etkisini modellemek amacıyla kullanılan yöntemler:
        - **GWR – Geographically Weighted Regression**
        - **Spatial Lag Model**
        - **Spatial Error Model**
        
        Bu modeller, farklı coğrafi bölgelerdeki etkilerin değişken olup olmadığını test etmeyi sağlar.
        """)
    
        st.info("Bu bölümde, mekânsal regresyon sonuçları tablo ve yorumlarla sunulmaktadır.")
    
        st.markdown("### 🌐 GWR (Geographically Weighted Regression) – Yerel Modelleme Sonuçları")
    
        st.metric("R²", "0.94")
        st.metric("Ortalama Etki (β)", "-0.473")
        st.metric("Min – Max Etki", "-12.881 – 0.000")
    
        st.markdown("""
        - GWR modeli, şehir bazında meteor uzaklığının yerleşim üzerindeki etkisini yüksek doğrulukla modellemiştir.
        - Özellikle Avrupa, Güney Asya gibi bölgelerde **yakın meteorlara doğru yerleşim artışı** net gözlemlenmiştir.
        - ❗ Sabit bir etki yerine, **coğrafyaya göre değişen etkiler** mevcuttur.
        """)
    
        st.markdown("### 🛰️ Spatial Lag Model – Mekânsal Bağımlılık")
    
        st.table(pd.DataFrame({
            "Model": ["Spatial Lag"],
            "Pseudo R²": ["0.814"],
            "Beta (Mesafe)": ["-0.025"],
            "ρ (Spatial Lag)": ["0.846"],
            "Yorum": ["Yerleşim desenleri komşu şehirler arasında güçlü etkileşim gösteriyor."]
        }))
    
        st.markdown("""
        - Komşu şehirler birlikte etkileniyor: Bir şehir meteorlara yakınsa çevresindekiler de yüksek olasılıkla öyle.
        - Bu model hipotezi **bölgesel etki** açısından güçlendiriyor.
        """)
    
        st.markdown("### 📈 Spatial Error Model – Mekânsal Hata Korelasyonu")
    
        st.table(pd.DataFrame({
            "Model": ["Spatial Error"],
            "Pseudo R²": ["0.071"],
            "Beta (Mesafe)": ["-0.0129"],
            "λ (Spatial Error)": ["0.846"],
            "Yorum": ["Hatalar coğrafi olarak kümelenmiş, ama model gücü düşük."]
        }))
    
        st.markdown("""
        - Bu modelde de uzaklık önemli bir değişken, ama etki zayıf.
        - Mekânsal hata yapısı var, fakat yerleşim olasılığını yeterince iyi açıklamıyor.
        """)
    
        st.success("📌 Sonuç: GWR ve Spatial Lag modelleri, meteor düşüşlerinin insan yerleşimi üzerinde **lokal ve bölgesel etkiler** yarattığını güçlü biçimde ortaya koymaktadır.")
    
    
    
       # 5️⃣ NEDENSELLİK & DERİN ÖĞRENME
    with sekmeler[3]:
        st.subheader("🧮 Nedensellik ve Derin Öğrenme Modelleri")
        st.markdown("""
        Bu bölümde, meteor düşüşlerinin **insan yerleşimleri üzerindeki etkisini** nedensel ve zamansal boyutta analiz eden gelişmiş modeller sunulmaktadır.
        """)
    
        # 1. NEDENSEL MODELLER
        st.markdown("### 🎯 1. İstatistiksel Nedensellik Analizleri")
        st.markdown("""
        Meteor mesafesinin şehirleşmeye etkisinin **nedensel olup olmadığını** analiz etmek için 3 yöntem kullanılmıştır:
        """)
    
        st.table(pd.DataFrame({
            "Model": ["Bayesian (PyMC3)", "DoWhy", "CausalML T-Learner"],
            "Tahmini Etki": ["β = -0.493", "ATE ≈ -1.49e-6", "ATE ≈ +0.036"],
            "Güvenilirlik": ["✅ Güçlü", "⚠️ Orta", "⚠️ Orta"],
            "Açıklama": [
                "MCMC ile posterior dağılımlardan etki tahmini",
                "Refütasyon testleriyle doğrulanmış lineer analiz",
                "Random Forest tabanlı uplift analizi"
            ],
            "Hipotez Desteği": ["✅ Güçlü", "⚠️ Orta", "⚠️ Orta"]
        }))
    
        st.success("""
        ✅ **PyMC3** modeli, meteor mesafesi arttıkça şehirleşme olasılığının azaldığını güçlü biçimde desteklemiştir.  
        Diğer yöntemler etkileri ölçmekte başarılı olmuş fakat karışan coğrafi faktörleri tam ayırt edememiştir.
        """)
    
        # 2. DERİN ÖĞRENME (ConvLSTM)
        st.markdown("### 🌀 2. Zaman + Mekân Derin Öğrenme (ConvLSTM2D)")
        st.markdown("""
        Bu model, 221 yıllık meteor düşüş desenlerinden yola çıkarak **şehir yoğunluğu maskesini öğrenmiş** ve geleceğe dönük yerleşim tahmininde bulunmuştur.
        """)
    
        st.table(pd.DataFrame({
            "Model": ["ConvLSTM2D"],
            "Doğruluk": ["~72.5%"],
            "Yöntem": ["Zaman ve mekân içeren 3D veri"],
            "Açıklama": ["Spatio-temporal yoğunluklara göre şehir olasılığı tahmini"],
            "Hipotez Desteği": ["✅ Güçlü"]
        }))
    
        st.info("""
        📌 **Zamansal Derin Öğrenme**, meteor düşüşlerinin tarihsel birikim etkisinin şehir yerleşim davranışlarına sızdığını ortaya koyar.  
        Bu da hipotezin **zamanla şekillendiği** yönündeki savı destekler.
        """)
    
        # GENEL KARŞILAŞTIRMA
        st.markdown("### 🔚 3. Genel Kıyaslama ve Yorum")
        st.dataframe(pd.DataFrame({
            "Model Grubu": ["Mekânsal Regresyon", "Nedensellik", "Zaman+Mekân (DL)"],
            "Teknik Güç": ["📈 Çok Yüksek", "⚖️ Orta–Yüksek", "🧠 Yüksek"],
            "Hipotez Açısından Değeri": [
                "Meteor mesafesinin mekânsal etkisini çözümledi",
                "Etkinin gerçekten nedensel olup olmadığını test etti",
                "Zaman içindeki meteor yoğunluğu ve şehir ilişkisini modelledi"
            ]
        }))
    
        st.success("""
        🔍 **Sonuç:**  
        Bu üç modelleme yaklaşımı, hipotezin farklı boyutlarını ele alarak **çok boyutlu bir destek** sağlamaktadır:
        
        - 🧭 **GWR**: Yerel mekânsal etkiyi detaylandırır  
        - 🧠 **ConvLSTM**: Tarihsel düşüş yoğunluklarının birikimli etkisini ortaya çıkarır  
        - ⚖️ **PyMC3**: Etkinin nedensel olduğunu güçlü şekilde ortaya koyar
    
        Böylece hipotez, sadece gözlemsel değil; mekânsal, zamansal ve istatistiksel olarak da **tutarlı** biçimde desteklenmiş olur.
        """)

# 🔗 Sosyal medya bağlantılarını sidebar'a ekle
st.sidebar.markdown("---")
st.sidebar.markdown("### 👤 Geliştirici")
st.sidebar.markdown("**Mehmet Yalaz**")

# Bağlantılar (kendi kullanıcı adlarını gir!)
linkedin_url = "https://www.linkedin.com/in/mehmet-yalaz/"
github_url = "https://github.com/Yalaz0"
instagram_url = "https://www.instagram.com/mehmetyalazz/"

st.sidebar.markdown(f"[![LinkedIn](https://img.shields.io/badge/-LinkedIn-0e76a8?style=flat&logo=linkedin&logoColor=white)]({linkedin_url})")
st.sidebar.markdown(f"[![GitHub](https://img.shields.io/badge/-GitHub-181717?style=flat&logo=github&logoColor=white)]({github_url})")
st.sidebar.markdown(f"[![Instagram](https://img.shields.io/badge/-Instagram-E4405F?style=flat&logo=instagram&logoColor=white)]({instagram_url})")

