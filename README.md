# Credit Scoring Model for Loan Approval Decisions

German Credit (Statlog) dataseti üzərində qurulmuş kredit skorinq modeli: müraciətçinin profilindən risk skoru çıxarır və həmin skoru əsaslandırılmış **approve / decline** qərarına çevirir.

Layihənin əsas məsələsi default proqnozu deyil — **kredit səhvlərinin asimmetrik xərcini əks etdirən qərar həddini (threshold) seçməkdir.**

---

## Dataset

| | |
|---|---|
| Mənbə | UCI Statlog (German Credit Data), Prof. Hans Hofmann, Hamburq Universiteti, 1994 |
| Həcm | 1000 kredit müraciəti, 20 feature |
| Target | `1` = good (700), `2` = bad (300) → bad rate **30%** |
| Missing value | Yoxdur |
| Valyuta | Deutsche Mark (avrodan əvvəlki dövr) |

Bütün 20 feature müraciət anında bankın əlində olur, ona görə **application-time** şərti avtomatik ödənilir və data leakage riski yoxdur.

### Cost matrix

Dataset sənədləşməsində verilmiş xərc matrisi bütün layihənin oxudur:

| | Good proqnoz | Bad proqnoz |
|---|---|---|
| **Faktiki Good** | 0 | **1** |
| **Faktiki Bad** | **5** | 0 |

Pis müştəriyə kredit vermək (FN), yaxşı müştərini rədd etməkdən (FP) **5 dəfə bahadır**. Məntiqi aydındır: FP-də bank faiz gəlirini itirir, FN-də isə kreditin əsas məbləğini.

### Fayllar

| Fayl | İzah | İstifadə |
|---|---|---|
| `german.data` | 20 feature + target, kateqoriyalar `A11`, `A34` kodları ilə | **Bəli** |
| `german.data-numeric` | Əvvəlcədən one-hot edilmiş 24 sütun | Xeyr — feature adları itib, SHAP izahı mənasız çıxır |
| `german.doc` | Data dictionary | Kodların açılması üçün |

---

## Metodologiya

### 1. Data hazırlığı

Kateqorik kodlar (`A11`, `A34`) oxunaqlı etiketlərə çevrilib (`lt_0_DM`, `critical_account`). Bu, iki yerdə qazandırır: EDA cədvəllərində və SHAP izahlarında. `.map()` tanımadığı kodu `NaN` etdiyi üçün mapping-in tamlığı `assert` ilə yoxlanılır.

Target `bad = 1` (pozitiv sinif) kimi kodlanıb. Bu qərar bütün metrikaların mənasını müəyyən edir: `recall` birbaşa "pis müştərilərin neçə faizini tutdum" deməkdir. Orijinal `target` sütunu silinib — saxlansaydı leakage yaradardı.

### 2. Train/test split

Split **EDA-dan əvvəl** aparılıb. Səbəb: bütün dataya baxıb qərar versəm (hansı feature-u atım, necə binləyim), test məlumatı analitik vasitəsilə modelə sızır. Stratified, `test_size=0.2` → 800 / 200, hər iki tərəfdə bad rate 30%.

### 3. EDA

Kredit skorinqdə əsas alət `value_counts()` deyil, hər kateqoriya üçün **bad rate** və portfel ortalamasına nisbətdə **lift**-dir.

**Ən güclü feature-lar (spread üzrə):** `credit_history` (0.438), `checking_status` (0.359), `savings` (0.242).

`purpose` yüksək spread verir, amma bu 5–10 nəfərlik kateqoriyalardan gəlir — səs-küydür. **Spread həmişə `count` ilə birlikdə oxunmalıdır.**

**İki intuisiyaya zidd tapıntı:**

- **`no_account` ən az riskli qrupdur (12%)**, halbuki ən riskli görünür. Bu adamlar əsas bankçılığını başqa bankda edir; bu banka overdraft üçün yox, sırf kredit üçün gəliblər. Train datasının ~40%-i budur — modelin ən böyük siqnalı.
- **`no_credits_all_paid` ən riskli qrupdur (61%)**, `critical_account` isə ən az riskli (17%). Bank tarixçəsi olmayanı qiymətləndirə bilmir (*thin file* problemi). Kritik hesabı olanlar isə artıq tanınan, aktiv kredit istifadəçiləridir.

**Numerik feature-lar:**

- `duration` monoton artır (4–12 ay ~20%, 30–60 ay ~46%) — ən təmiz numerik siqnal
- `credit_amount` **monoton deyil**, U formasındadır: həm çox kiçik, həm çox böyük kreditlər riskli. Logistic Regression xətti əlaqə fərz etdiyi üçün bu nümunəni tam tuta bilmir, XGBoost tutur — iki model arasındakı fərqin bir hissəsi buradan gəlir
- `duration` ↔ `credit_amount` korrelyasiya **0.64** — kritik səviyyə deyil, amma LogReg-də bu iki əmsalın fərdi şərhi etibarsızlaşır
- `credit_amount` skew **1.83** → xətti model üçün scaling lazımdır, ağac modelləri üçün yox

### 4. Preprocessing

İki ayrı `ColumnTransformer` qurulub, çünki iki model fərqli giriş istəyir:

| | Logistic Regression | XGBoost |
|---|---|---|
| Scaling | `StandardScaler` — regularizasiya bütün feature-ları eyni miqyasda cəzalandırır | `passthrough` — bölmə nöqtəsi miqyasdan asılı deyil |
| One-hot | `drop="first"` — dummy trap / mükəmməl kollinearlıq | Drop edilmir, SHAP-ın tam qalması üçün |

`min_frequency=20` nadir kateqoriyaları avtomatik birləşdirir. Əl ilə birləşdirməkdən üstündür, çünki qərar train-ə əsasən verilir və cross-validation-un **hər fold-unda yenidən** hesablanır.

Bütün preprocessing `Pipeline` içindədir. `pd.get_dummies` ilə əl işi üç problem yaradardı: (1) scaler statistikasının test-dən sızması, (2) CV-də hər fold üçün yenidən fit oluna bilməməsi, (3) train və test-də fərqli sütun sayı.

### 5. Modellər

| Model | OOF AUC | PR-AUC | Brier |
|---|---|---|---|
| **XGBoost (tuned)** | **0.7961** | **0.6063** | **0.1616** |
| XGBoost | 0.7872 | 0.5989 | 0.1674 |
| XGBoost + `scale_pos_weight` | 0.7865 | 0.5988 | 0.1768 |
| Logistic Regression | 0.7748 | 0.5945 | 0.1710 |
| LogReg + `class_weight="balanced"` | 0.7737 | 0.5899 | 0.1934 |

Hiperparametrlər `RandomizedSearchCV` (40 iterasiya, 5-fold) ilə seçilib. Ağac dərinliyi aşağı saxlanılıb — 800 sətirlik datasetdə overfit riski yüksəkdir.

### 6. Imbalance: niyə threshold, niyə model daxilində yox

`scale_pos_weight` və `class_weight="balanced"` **sıralama gücünü dəyişmir** (AUC 0.7872 → 0.7865), amma **ehtimalları yuxarı sürükləyir** — Brier 0.1674 → 0.1768.

Nəticə: model hər kəsi daha riskli göstərir, ehtimal artıq "həmin adamın real default şansı" demək deyil. Bu, scorecard və kapital hesablaması üçün problemdir.

Ona görə bu layihədə imbalance **modelin daxilində deyil, threshold səviyyəsində** həll olunub. Ehtimalların mənası qorunur, asimmetrik xərc isə qərar qaydasına köçürülür.

### 7. Threshold seçimi

Default 0.5 **simmetrik xəta xərci** fərz edir — burada FN 5 dəfə bahadır, ona görə 0.5 səhv defoltdur.

Nəzəri optimal threshold:

```
t* = C_FP / (C_FP + C_FN) = 1 / (1 + 5) ≈ 0.167
```

Empirik optimum **out-of-fold ehtimallar** üzərində axtarılıb (test dəstinə toxunmadan): **0.174**.

İki rəqəmin üst-üstə düşməsi təsadüf deyil — modelin ehtimallarının məqbul dərəcədə kalibr olunduğunun təsdiqidir. Xərc əyrisi optimum ətrafında yastıdır, yəni threshold-un kiçik dəyişməsi nəticəni kökündən pozmur.

---

## Nəticələr (test dəsti, n = 200, `SEED = 42`)

**Test AUC 0.7904 · PR-AUC 0.6346**

| Qərar qaydası | Threshold | TP | FP | FN | TN | Recall (bad) | Approve % | **COST** |
|---|---|---|---|---|---|---|---|---|
| Default | 0.500 | 30 | 20 | 30 | 120 | 0.500 | 75.0% | 170 |
| Nəzəri | 0.167 | 50 | 67 | 10 | 73 | 0.833 | 41.5% | **117** |
| **Seçilən** | **0.174** | 50 | 67 | 10 | 73 | **0.833** | 41.5% | **117** |

Model eynidir — dəyişən yalnız qərar qaydasıdır. Xərc **31% azalıb**, recall iki dəfədən çox artıb.

> **Accuracy niyə hesabatda yoxdur:** "hamıya approve" desəm 70% accuracy alıb 300 vahid xərc yaradaram. Bu problemdə accuracy mənasız metrikadır; optimallaşdırılan kəmiyyət **cost**-dur.

---

## Explainability (SHAP)

Tənzimləyici və müştəri qarşısında "model belə dedi" cavab deyil. Hər rədd qərarı üçün konkret səbəb göstərilə bilməlidir.

SHAP dəyərləri **orijinal feature səviyyəsinə yığılıb** — bir feature-dan yaranan bütün one-hot sütunların töhfəsi toplanıb. Bu addım olmasa səbəb siyahısında `checking_status_no_account riski artırdı` kimi mənasız sətirlər çıxır (müştəridə həmin kateqoriya yoxdur, SHAP sadəcə onun **olmamasını** cəzalandırır).

Nümunə — ən yüksək riskli rədd edilmiş müraciətçi (p = 0.914):

```
checking_status   = lt_0_DM               riski +0.827
credit_history    = no_credits_all_paid   riski +0.671
property          = no_property           riski +0.418
duration          = 48                    riski +0.377
```

Bu format birbaşa **adverse action notice**-ə çevrilə bilər.

Qlobal SHAP nəticələri EDA ilə üst-üstə düşür (`no_account` riski azaldır, uzun `duration` artırır) — modelin sağlam öyrəndiyinin göstəricisidir.

---

## Scorecard (bonus)

Bank ehtimalla yox, **xalla** işləyir. Standart çevirmə logit üzərində xəttidir:

```
factor = PDO / ln(2)
offset = base_score − factor × ln(base_odds)
score  = offset + factor × ln(odds_good)
```

Sənayedə adətən "600 xal = 50:1 odds" reperi götürülür, amma o, bad rate-i ~2% olan portfel üçündür. Bu portfelin bad rate-i 30%-dir (odds ~2.33:1), ona görə reper **portfelin öz ortalamasına** bağlanıb: orta müştəri 600 xal alır.

`PDO = 20`, `base_score = 600`, `base_odds = 2.33` → **cutoff = 620 xal**

| Score bandı | Müştəri sayı | Faktiki bad rate |
|---|---|---|
| < 550 | 23 | **78.3%** |
| 550–580 | 33 | 42.4% |
| 580–610 | 45 | 37.8% |
| 610–640 | 45 | 15.6% |
| 640+ | 54 | **7.4%** |

Bandlar **tam monotondur** — scorecard düzgün işləyir.

---

## Kalibrasiya (bonus)

| | Brier | AUC |
|---|---|---|
| Xam | 0.1627 | 0.7904 |
| İsotonic | 0.1621 | 0.7890 |

Qazanc cüzidir və bu, olduğu kimi göstərilir. Səbəbi aydındır: `scale_pos_weight` istifadə edilmədiyi üçün model onsuz da nisbətən yaxşı kalibr olunub — düzəltməli çox şey qalmayıb. AUC-un dəyişməməsi gözləniləndir, çünki isotonic regression monotondur: sıralamanı saxlayır, sadəcə dəyərləri sürüşdürür.

800 sətirlik datasetdə isotonic özü də overfit riski daşıyır; bu ölçü üçün `method="sigmoid"` (Platt) daha davamlı seçim ola bilərdi.

Əsas nəticə **müqayisədədir**: imbalance `scale_pos_weight` ilə həll edilsəydi, Brier ciddi pisləşərdi və kalibrasiya məcburi olardı.

---

## Fairness (bonus)

| Cins | n | Faktiki bad rate | Approve rate | TPR | **FPR** |
|---|---|---|---|---|---|
| Qadın | 61 | 0.328 | 0.311 | 0.850 | **0.610** |
| Kişi | 139 | 0.288 | 0.460 | 0.825 | **0.424** |

| Yaş qrupu | n | Faktiki bad rate | Approve rate | TPR | FPR |
|---|---|---|---|---|---|
| Gənc (<26) | 44 | 0.455 | 0.250 | 0.800 | 0.708 |
| Yetkin | 156 | 0.256 | 0.462 | 0.850 | 0.431 |

**Şərh:**

- **Approve rate fərqi özlüyündə ayrı-seçkilik demək deyil** — qruplarda faktiki bad rate də fərqlidirsə, model reallığı əks etdirir.
- **FPR fərqi isə ciddi problemdir**: eyni dərəcədə yaxşı müştərilər qadın qrupunda daha çox rədd olunur (0.610 vs 0.424). Bu, həm hüquqi, həm kommersiya baxımından itkidir.

**Cins feature-u çıxarıldıqda:**

| | AUC | Cost |
|---|---|---|
| `personal_status_sex` ilə | 0.7904 | 117 |
| Onsuz | 0.8052 | 109 |

Model nə AUC, nə də cost baxımından itirmir (hətta bir qədər yaxşılaşır, amma 200 sətirlik test dəstində bu fərq səs-küy hüdudundadır). Nəticə: bu feature-u saxlamaq üçün kommersiya əsası yoxdur, hüquqi risk isə realdır — **çıxarılmalıdır**.

**Metodoloji məhdudiyyət:** `personal_status_sex` cinslə ailə statusunu birləşdirir. `A95` (female single) datada ümumiyyətlə yoxdur, ona görə cins ayrıla bilir, amma qadınlar üçün ailə statusu ayrılmayıb, kişilər üçün ayrılıb. Yəni iki qrupu müqayisə edəndə cins effekti ilə ailə statusu effekti qismən bir-birinə qarışır.

---

## Məhdudiyyətlər

- **Dövr.** Data 1994-cü ilin Alman bank portfelidir, Deutsche Mark ilə. Bugünkü şəraitə birbaşa köçürülməz.
- **Həcm.** Test dəsti 200 nəfərdir; alt qruplar (gənc qadınlar — 20-yə yaxın müşahidə) üzrə nəticələr dar nümunəyə söykənir və geniş etibarlılıq intervalı ilə oxunmalıdır.
- **Reject inference edilməyib.** Dataset yalnız **onaylanmış** müraciətçiləri əhatə edir — rədd edilənlərin faktiki davranışı bilinmir. Real bankda bu, ciddi selection bias yaradır və modelin qiymətləndirməsini optimist edir.
- **Cost matrix sabitdir.** 5:1 nisbəti bütün müraciətçilər üçün eyni götürülüb; realda kredit məbləğinə görə dəyişir. Məbləğə əsaslanan xərc funksiyası daha dəqiq olardı.
- **Gəlir tərəfi modelləşdirilməyib.** Seçilən threshold approve rate-i ~41%-ə salır ki, real bankda həcm baxımından qəbul edilməzdir. 5:1 matrisi yalnız **itki** tərəfini əks etdirir. Tam məqsəd funksiyası `gəlir(TN) − itki(FN) − itirilmiş_gəlir(FP)` şəklində qurulmalıdır.

---

## Struktur və işə salma

```
.
├── credit_scoring.ipynb    # əsas notebook
├── german.data             # dataset (UCI-dan yüklənir)
├── german.doc              # data dictionary
└── README.md
```

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
jupyter notebook credit_scoring.ipynb
```

`german.data` faylı notebook ilə eyni qovluqda olmalıdır.

**Reproduksiya:** `SEED = 42`. Bu README-dəki bütün rəqəmlər həmin seed ilə alınıb; fərqli seed-də dəyərlər bir qədər fərqlənəcək, amma istiqamətlər dəyişmir.

---

## Texniki stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap` · `matplotlib`
