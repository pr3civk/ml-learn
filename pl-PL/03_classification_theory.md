### **Wstęp (Rozdziały 1 i 2 - tło)**

*   **Streszczenie:** Autor przypomina, że uczenie maszynowe nadzorowane dzieli się na dwa główne typy zadań: **regresję** (przewidywanie ciągłych wartości, np. ceny domu) i **klasyfikację** (przewidywanie kategorii/klas, np. czy na zdjęciu jest cyfra "5"). Po omówieniu regresji w poprzednim rozdziale, teraz skupimy się na klasyfikacji.

---

### **Podrozdział: MNIST**

*   **Streszczenie:** W tym rozdziale będziemy używać zbioru danych MNIST. Jest to zbiór 70 000 małych, czarno-białych obrazków odręcznie pisanych cyfr (od 0 do 9). Każdy obrazek ma etykietę mówiącą, jaką cyfrę przedstawia. MNIST jest tak popularny, że nazywa się go "hello world" uczenia maszynowego - to standardowy problem, na którym testuje się nowe algorytmy klasyfikacji.

*   **Pojęcia i Listingi Kodu:**

    1.  **Pobieranie Danych**
        *   **Kod:**
            ```python
            from sklearn.datasets import fetch_openml
            mnist = fetch_openml('mnist_784', version=1)
            mnist.keys()
            ```
        *   **Teoria i Przeznaczenie:**
            *   `fetch_openml`: To funkcja z biblioteki Scikit-Learn, która służy do łatwego pobierania popularnych zbiorów danych. Zamiast szukać pliku w internecie, pobierać go i wczytywać ręcznie, ta jedna linijka robi wszystko za nas.
            *   `mnist.keys()`: Pobrane dane mają strukturę podobną do słownika w Pythonie. Użycie `.keys()` pozwala nam zobaczyć, co się w nim znajduje: `'data'` (dane, czyli obrazki), `'target'` (etykiety, czyli poprawne cyfry), `'DESCR'` (opis zbioru) itd. To pierwszy krok, żeby zorientować się, jak dobrać się do potrzebnych informacji.

    2.  **Struktura Danych (Zmienne X i y)**
        *   **Kod:**
            ```python
            X, y = mnist["data"], mnist["target"]
            X.shape # (70000, 784)
            y.shape # (70000,)
            ```
        *   **Teoria i Przeznaczenie:**
            *   W uczeniu maszynowym przyjęło się, że **`X`** to nasze dane wejściowe (cechy), a **`y`** to etykiety (poprawne odpowiedzi), których model ma się nauczyć. Robimy to, aby oddzielić "problem" od "rozwiązania".
            *   `X.shape` pokazuje nam wymiary danych: 70 000 wierszy (obrazków) i 784 kolumny (cechy). Każdy obrazek ma 28x28 pikseli, co daje 784 piksele. Każdy piksel to jedna cecha, opisująca jego jasność (od 0 do 255).
            *   `y.shape` pokazuje, że mamy 70 000 etykiet, po jednej dla każdego obrazka.

    3.  **Wizualizacja Danych**
        *   **Kod:**
            ```python
            import matplotlib.pyplot as plt
            some_digit = X[0]
            some_digit_image = some_digit.reshape(28, 28)
            plt.imshow(some_digit_image, cmap="binary")
            plt.axis("off")
            plt.show()
            ```
        *   **Teoria i Przeznaczenie:**
            *   Dane w `X` są "spłaszczone" do jednowymiarowej listy 784 liczb. Aby zobaczyć obrazek, musimy przywrócić mu jego oryginalny, dwuwymiarowy kształt (28x28 pikseli). Do tego służy funkcja `.reshape(28, 28)`.
            *   **Dlaczego to robimy?** Zawsze warto spojrzeć na dane, aby nabrać intuicji. Sprawdzamy, czy dane wyglądają sensownie i czy etykieta (`y[0]`, która okazuje się być '5') pasuje do tego, co widzimy na obrazku.

    4.  **Przygotowanie Danych (Casting)**
        *   **Kod:**
            ```python
            y = y.astype(np.uint8)
            ```
        *   **Teoria i Przeznaczenie:**
            *   Etykiety `y` były początkowo wczytane jako tekst (np. '5'). Większość algorytmów uczenia maszynowego działa na liczbach, a nie na tekście. Ta linijka zamienia tekstowe etykiety na liczby całkowite. Jest to typowy krok czyszczenia i przygotowywania danych.

    5.  **Podział na Zbiór Treningowy i Testowy**
        *   **Kod:**
            ```python
            X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
            ```
        *   **Teoria i Przeznaczenie:**
            *   To **jedna z najważniejszych zasad** w uczeniu maszynowym. Dzielimy cały nasz zbiór danych na dwie części: **treningową** (`_train`) i **testową** (`_test`).
            *   **Zbiór treningowy** służy do "uczenia" modelu. Model patrzy na te dane i próbuje znaleźć w nich wzorce.
            *   **Zbiór testowy** jest jak egzamin końcowy. Trzymamy go w ukryciu i używamy **tylko raz, na samym końcu**, aby sprawdzić, jak dobrze nasz ostateczny model radzi sobie z danymi, których **nigdy wcześniej nie widział**.
            *   **Dlaczego to takie ważne?** Gdybyśmy testowali model na tych samych danych, na których się uczył, mógłby on po prostu "zapamiętać" odpowiedzi na pamięć, zamiast nauczyć się generalizować. Taki model miałby świetne wyniki na danych treningowych, ale byłby bezużyteczny w prawdziwym świecie, bo nie potrafiłby poradzić sobie z nowymi danymi. Ten podział pozwala nam uczciwie ocenić, czy model faktycznie się czegoś nauczył.

---

### **Podrozdział: Trening Klasyfikatora Binarnego**

*   **Streszczenie:** Zamiast od razu próbować rozpoznawać wszystkie 10 cyfr, upraszczamy problem. Tworzymy klasyfikator, który odpowiada tylko na jedno pytanie: "Czy ten obrazek to cyfra 5?". Taki klasyfikator, rozróżniający tylko dwie klasy (w tym przypadku: "5" i "nie-5"), nazywa się **klasyfikatorem binarnym**. Do tego celu użyjemy modelu `SGDClassifier`.

*   **Pojęcia i Listingi Kodu:**

    1.  **Tworzenie Etykiet dla Klasyfikatora Binarnego**
        *   **Kod:**
            ```python
            y_train_5 = (y_train == 5) # True dla wszystkich 5, False dla reszty
            y_test_5 = (y_test == 5)
            ```
        *   **Teoria i Przeznaczenie:**
            *   Musimy dostosować nasze etykiety do nowego, prostszego problemu. Tworzymy nowe zmienne `y`, które zawierają wartość `True` (prawda), jeśli oryginalna cyfra to 5, i `False` (fałsz) w każdym innym przypadku. Model będzie teraz uczył się przewidywać jedną z tych dwóch wartości.

    2.  **Wybór i Trening Modelu**
        *   **Kod:**
            ```python
            from sklearn.linear_model import SGDClassifier
            sgd_clf = SGDClassifier(random_state=42)
            sgd_clf.fit(X_train, y_train_5)
            ```
        *   **Teoria i Przeznaczenie:**
            *   `SGDClassifier` (Stochastic Gradient Descent Classifier) to prosty, ale wydajny model. Tekst wspomina, że dobrze radzi sobie z dużymi zbiorami danych, ponieważ przetwarza dane partiami lub nawet pojedynczo, co oszczędza pamięć.
            *   `random_state=42`: SGD ma w sobie element losowości. Ustawienie `random_state` na konkretną liczbę (np. 42) sprawia, że ta "losowość" jest zawsze taka sama. **Dlaczego to robimy?** Dzięki temu nasze eksperymenty są **powtarzalne**. Jeśli ktoś inny uruchomi nasz kod, dostanie dokładnie ten sam wynik, co ułatwia współpracę i weryfikację.
            *   `sgd_clf.fit(X_train, y_train_5)`: To jest moment "uczenia się". Mówimy modelowi (`sgd_clf`), żeby przeanalizował dane treningowe (`X_train`) i ich binarne etykiety (`y_train_5`) i znalazł reguły pozwalające odróżnić cyfrę 5 od pozostałych.

    3.  **Dokonywanie Predykcji**
        *   **Kod:**
            ```python
            sgd_clf.predict([some_digit])
            ```
        *   **Teoria i Przeznaczenie:**
            *   Po wytrenowaniu model jest gotowy do pracy. Metoda `.predict()` służy do przewidywania etykiety dla nowych danych. Dajemy mu obrazek (`some_digit`), a on zwraca swoją prognozę (w tym przypadku `True`, co oznacza, że rozpoznał cyfrę 5).

---

### **Podrozdział: Miary Wydajności (Performance Measures)**

*   **Streszczenie:** Samo stwierdzenie, że model działa, nie wystarczy. Musimy zmierzyć, **jak dobrze** działa. Ocena klasyfikatora jest bardziej skomplikowana niż ocena regresora. Poznamy różne metryki, takie jak dokładność, macierz pomyłek, precyzja i czułość, oraz dowiemy się, dlaczego prosta dokładność może być myląca.

*   **Pojęcia i Listingi Kodu:**

    1.  **Walidacja Krzyżowa (Cross-Validation)**
        *   **Kod:**
            ```python
            from sklearn.model_selection import cross_val_score
            cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
            ```
        *   **Teoria i Przeznaczenie:**
            *   Zamiast jednorazowego podziału na zbiór treningowy i testowy (który zostawiamy na koniec), możemy ocenić model bardziej wiarygodnie za pomocą **walidacji krzyżowej**.
            *   **Jak to działa?** `cv=3` oznacza, że zbiór treningowy jest dzielony na 3 równe części (tzw. "foldy"). Model jest trenowany na dwóch częściach, a testowany na trzeciej. Proces ten powtarza się 3 razy, za każdym razem używając innej części do testowania.
            *   **Dlaczego to robimy?** Dostajemy kilka ocen zamiast jednej. Daje nam to znacznie lepszy obraz stabilności i rzeczywistej wydajności modelu. Jeśli wyniki na wszystkich foldach są podobne i wysokie, jesteśmy bardziej pewni, że model jest dobry. Pojedynczy podział mógł być przypadkowo "szczęśliwy" lub "pechowy".
            *   `scoring="accuracy"`: Mówimy, jaką metryką chcemy ocenić model. Tutaj jest to **dokładność** (accuracy).

    2.  **Pułapka Dokładności (Accuracy)**
        *   **Kod:**
            ```python
            class Never5Classifier(BaseEstimator):
                # ...
            cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
            # Wynik: ponad 90%
            ```
        *   **Teoria i Przeznaczenie:**
            *   Autor tworzy "głupi" klasyfikator, który **zawsze** twierdzi, że obrazek **nie jest** cyfrą 5.
            *   **Problem:** Cyfry "5" stanowią tylko około 10% zbioru danych. Zatem klasyfikator, który zawsze zgaduje "nie-5", będzie miał rację w 90% przypadków! Jego dokładność wyniesie ponad 90%, co brzmi fantastycznie, ale w rzeczywistości model jest bezużyteczny.
            *   **Wniosek:** Dokładność jest bardzo złą metryką, gdy mamy do czynienia ze **zbiorami niezbalansowanymi** (gdzie jedna klasa występuje znacznie częściej niż inne).

    3.  **Macierz Pomyłek (Confusion Matrix)**
        *   **Kod:**
            ```python
            from sklearn.model_selection import cross_val_predict
            y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

            from sklearn.metrics import confusion_matrix
            confusion_matrix(y_train_5, y_train_pred)
            ```
        *   **Teoria i Przeznaczenie:**
            *   `cross_val_predict`: Działa podobnie do `cross_val_score`, ale zamiast zwracać oceny, zwraca **predykcje** dla każdej próbki ze zbioru treningowego. Każda predykcja jest robiona przez model, który nie widział tej konkretnej próbki podczas swojego treningu. Dzięki temu otrzymujemy "czyste" predykcje dla całego zbioru treningowego.
            *   `confusion_matrix`: To tabela, która pokazuje, jak model się mylił. Ma 4 kluczowe pola:
                *   **True Negative (TN):** Poprawnie sklasyfikowane "nie-5".
                *   **False Positive (FP):** "Nie-5", które model błędnie oznaczył jako "5". (Fałszywy alarm)
                *   **False Negative (FN):** "5", której model nie wykrył. (Przeoczenie)
                *   **True Positive (TP):** Poprawnie sklasyfikowane "5".
            *   **Dlaczego jej używamy?** Macierz pomyłek daje pełen obraz wydajności, pokazując nie tylko ile było poprawnych odpowiedzi, ale także **jakiego rodzaju błędy** popełniał model.

    4.  **Precyzja i Czułość (Precision & Recall)**
        *   **Kod:**
            ```python
            from sklearn.metrics import precision_score, recall_score
            precision_score(y_train_5, y_train_pred)
            recall_score(y_train_5, y_train_pred)
            ```
        *   **Teoria i Przeznaczenie:**
            *   **Precyzja (Precision):** Odpowiada na pytanie: "Jaki procent przykładów, które model oznaczył jako '5', to faktycznie były '5'?". Wysoka precyzja oznacza, że jeśli model coś twierdzi, to można mu ufać. Jest ważna, gdy koszt **fałszywego alarmu (FP)** jest wysoki (np. klasyfikowanie bezpiecznych filmów dla dzieci jako niebezpieczne).
                *   *Wzór:* `TP / (TP + FP)`
            *   **Czułość (Recall/Sensitivity):** Odpowiada na pytanie: "Jaki procent wszystkich '5', które były w zbiorze, model faktycznie znalazł?". Wysoka czułość oznacza, że model niewiele przeocza. Jest ważna, gdy koszt **przeoczenia (FN)** jest wysoki (np. nie wykrycie chorej osoby lub złodzieja w sklepie).
                *   *Wzór:* `TP / (TP + FN)`

    5.  **Wynik F1 (F1 Score)**
        *   **Kod:**
            ```python
            from sklearn.metrics import f1_score
            f1_score(y_train_5, y_train_pred)
            ```
        *   **Teoria i Przeznaczenie:**
            *   To pojedyncza metryka, która łączy precyzję i czułość w jedną liczbę. Jest to **średnia harmoniczna**, co oznacza, że daje wysoką wartość tylko wtedy, gdy **obie** metryki (precyzja i czułość) są wysokie. Jest to użyteczne, gdy szukamy zrównoważonego klasyfikatora.

    6.  **Kompromis Precyzja/Czułość (Precision/Recall Trade-off)**
        *   **Kod:**
            ```python
            y_scores = sgd_clf.decision_function([some_digit])
            threshold = 8000
            y_some_digit_pred = (y_scores > threshold)
            ```
        *   **Teoria i Przeznaczenie:**
            *   Niestety, prawie nigdy nie da się mieć 100% precyzji i 100% czułości jednocześnie. Zwiększanie jednej z tych metryk często powoduje spadek drugiej.
            *   **Jak to działa?** Model nie zwraca po prostu "tak/nie". Wewnętrznie oblicza "wynik" (score), który mówi, jak bardzo jest pewien swojej decyzji. Domyślnie, jeśli wynik jest powyżej progu 0, klasyfikuje jako "5".
            *   Możemy ten **próg (threshold)** zmieniać.
                *   **Wysoki próg:** Model będzie klasyfikował jako "5" tylko te obrazki, co do których jest *bardzo* pewien. To zwiększy **precyzję** (mniej fałszywych alarmów), ale zmniejszy **czułość** (więcej przeoczeń).
                *   **Niski próg:** Model będzie bardziej "liberalny" i oznaczy jako "5" więcej obrazków. To zwiększy **czułość** (mniej przeoczeń), ale zmniejszy **precyzję** (więcej fałszywych alarmów).
            *   `decision_function()`: Zamiast prosić o ostateczną predykcję (`.predict()`), możemy poprosić o ten wewnętrzny "wynik" (`.decision_function()`). Daje nam to kontrolę nad progiem i pozwala dostosować model do naszych potrzeb.

    7.  **Krzywa ROC**
        *   **Kod:**
            ```python
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
            ```
        *   **Teoria i Przeznaczenie:**
            *   Krzywa ROC to kolejny sposób na ocenę i wizualizację klasyfikatora binarnego. Pokazuje ona zależność między **odsetkiem prawdziwie pozytywnych (TPR)**, co jest inną nazwą na czułość, a **odsetkiem fałszywie pozytywnych (FPR)** dla różnych progów decyzyjnych.
            *   **FPR** to odsetek negatywnych przykładów, które zostały błędnie sklasyfikowane jako pozytywne.
            *   **Jak ją czytać?** Idealny klasyfikator ma krzywą, która biegnie jak najbliżej lewego górnego rogu (100% TPR, 0% FPR). Linia przerywana po przekątnej reprezentuje losowy klasyfikator.
            *   **AUC (Area Under Curve):** Pole pod krzywą ROC. To pojedyncza liczba podsumowująca jakość modelu. 1.0 oznacza idealny klasyfikator, a 0.5 oznacza klasyfikator losowy (bezużyteczny).
            *   **Kiedy używać ROC, a kiedy Precision/Recall?** Tekst sugeruje regułę: używaj krzywej **Precision/Recall**, gdy klasa pozytywna jest rzadka lub gdy bardziej zależy Ci na fałszywych alarmach (FP). W innych przypadkach krzywa ROC jest dobrym wyborem.

---

### **Podrozdział: Klasyfikacja Wieloklasowa (Multiclass Classification)**

*   **Streszczenie:** Przechodzimy od problemu binarnego ("5" czy "nie-5") do oryginalnego problemu rozpoznawania wszystkich 10 cyfr (0-9). Niektóre algorytmy, jak `SGDClassifier` czy `RandomForestClassifier`, potrafią to robić naturalnie. Inne, jak `Support Vector Machine`, są z natury binarne i wymagają specjalnych strategii, aby poradzić sobie z wieloma klasami.

*   **Pojęcia i Listingi Kodu:**

    1.  **Strategie Klasyfikacji Wieloklasowej**
        *   **Teoria i Przeznaczenie:**
            *   **One-versus-the-Rest (OvR) / One-versus-All (OvA):** Trenujemy 10 osobnych klasyfikatorów binarnych: jeden do odróżniania "0" od reszty, drugi do odróżniania "1" od reszty, itd. Aby sklasyfikować nowy obrazek, przepuszczamy go przez wszystkie 10 klasyfikatorów i wybieramy tę klasę, której klasyfikator dał najwyższy wynik.
            *   **One-versus-One (OvO):** Trenujemy klasyfikator dla każdej pary cyfr: jeden do odróżniania "0" od "1", drugi dla "0" i "2", kolejny dla "1" i "2" itd. W sumie dla 10 cyfr dałoby to 45 klasyfikatorów. Aby sklasyfikować nowy obrazek, przepuszczamy go przez wszystkie 45 klasyfikatorów i sprawdzamy, która klasa "wygrała" najwięcej pojedynków.
            *   **Scikit-Learn robi to automatycznie:** Kiedy używamy binarnego klasyfikatora (np. `SVC`) do zadania wieloklasowego, Scikit-Learn sam wybiera odpowiednią strategię (dla SVC jest to OvO).

    2.  **Klasyfikacja Wieloklasowa w Praktyce**
        *   **Kod:**
            ```python
            from sklearn.svm import SVC
            svm_clf = SVC()
            svm_clf.fit(X_train, y_train) # Używamy oryginalnych etykiet y_train
            svm_clf.predict([some_digit])
            ```
        *   **Teoria i Przeznaczenie:**
            *   Wystarczy użyć oryginalnych etykiet `y_train` (zawierających cyfry 0-9), a Scikit-Learn zajmie się resztą.
            *   `some_digit_scores = svm_clf.decision_function([some_digit])`: Dla zadania wieloklasowego `decision_function` zwraca 10 wyników - po jednym dla każdej klasy. Wybierana jest klasa z najwyższym wynikiem.

    3.  **Skalowanie Danych**
        *   **Kod:**
            ```python
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
            cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
            ```
        *   **Teoria i Przeznaczenie:**
            *   Wiele algorytmów uczenia maszynowego działa znacznie lepiej, gdy cechy (w tym przypadku wartości pikseli) mają podobny zakres (np. średnia 0 i odchylenie standardowe 1). `StandardScaler` to narzędzie, które przekształca dane do takiej postaci.
            *   **Dlaczego to robimy?** Poprawia to zbieżność i wydajność algorytmów (zwłaszcza tych opartych na gradiencie, jak SGD). W tym przykładzie skalowanie danych podniosło dokładność z ~84% do ~89%. Jest to bardzo ważny i często stosowany krok w przygotowaniu danych.

---

### **Podrozdział: Analiza Błędów (Error Analysis)**

*   **Streszczenie:** Mamy już działający model, ale chcemy go ulepszyć. Zamiast próbować na ślepo, najpierw analizujemy, **jakiego rodzaju błędy** on popełnia. To pozwala nam skupić wysiłki tam, gdzie przyniosą najwięcej korzyści.

*   **Pojęcia i Listingi Kodu:**

    1.  **Wizualizacja Macierzy Pomyłek**
        *   **Kod:**
            ```python
            y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
            conf_mx = confusion_matrix(y_train, y_train_pred)
            plt.matshow(conf_mx, cmap=plt.cm.gray)
            ```
        *   **Teoria i Przeznaczenie:**
            *   Tworzymy macierz pomyłek dla problemu wieloklasowego. Będzie to macierz 10x10.
            *   Wizualizacja `plt.matshow` pozwala szybko zorientować się w wynikach. Jasne pola na głównej przekątnej oznaczają poprawnie sklasyfikowane próbki. Ciemniejsze pola na przekątnej mogą sugerować, że dana klasa jest rzadsza lub model gorzej sobie z nią radzi. Jasne pola poza przekątną wskazują na częste pomyłki.

    2.  **Analiza Względnych Błędów**
        *   **Kod:**
            ```python
            row_sums = conf_mx.sum(axis=1, keepdims=True)
            norm_conf_mx = conf_mx / row_sums
            np.fill_diagonal(norm_conf_mx, 0)
            plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
            ```
        *   **Teoria i Przeznaczenie:**
            *   Chcemy skupić się na błędach, a nie na poprawnych predykcjach. Dzielimy każdą wartość w macierzy przez sumę wiersza, aby uzyskać **odsetek błędów** zamiast ich bezwzględnej liczby.
            *   **Dlaczego to robimy?** Unikamy sytuacji, w której klasy z dużą liczbą przykładów wyglądają na takie, w których model popełnia więcej błędów, tylko dlatego, że jest więcej okazji do pomyłki. Normalizacja pokazuje względne wskaźniki błędów.
            *   `np.fill_diagonal(norm_conf_mx, 0)`: Zerujemy główną przekątną, aby na wykresie zostały **tylko błędy**.
            *   **Wnioski z analizy:** Jasne komórki pokazują, które klasy są ze sobą mylone (np. cyfry 3 i 5). Jasna kolumna "8" oznacza, że wiele różnych cyfr jest błędnie klasyfikowanych jako "8". Taka analiza daje konkretne wskazówki, np. "musimy pomóc modelowi lepiej odróżniać cyfry od ósemek".

---

### **Podrozdział: Klasyfikacja Wieloelementowa (Multilabel Classification)**

*   **Streszczenie:** Do tej pory każdy obrazek mógł należeć tylko do jednej klasy. Czasami chcemy, aby model przypisał **wiele etykiet** do jednej próbki. Na przykład, na zdjęciu może być wiele osób, a system powinien rozpoznać każdą z nich.

*   **Pojęcia i Listingi Kodu:**

    *   **Kod:**
        ```python
        y_train_large = (y_train >= 7) # Etykieta 1: czy cyfra jest duża?
        y_train_odd = (y_train % 2 == 1) # Etykieta 2: czy cyfra jest nieparzysta?
        y_multilabel = np.c_[y_train_large, y_train_odd]

        from sklearn.neighbors import KNeighborsClassifier
        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train, y_multilabel)
        knn_clf.predict([some_digit]) # Zwraca np. [[False, True]]
        ```
    *   **Teoria i Przeznaczenie:**
        *   Tworzymy system, który dla każdej cyfry przewiduje dwie binarne etykiety: czy jest duża (7, 8, lub 9) ORAZ czy jest nieparzysta.
        *   `y_multilabel` jest teraz tablicą, gdzie każdy wiersz ma dwie kolumny z wartościami `True`/`False`.
        *   Nie wszystkie klasyfikatory to potrafią, ale `KNeighborsClassifier` tak.
        *   **Wynik:** Dla cyfry "5", model poprawnie przewiduje `[False, True]` (nie jest duża, jest nieparzysta).
        *   **Ocena:** Można mierzyć wynik F1 dla każdej etykiety osobno i uśrednić wyniki (`average="macro"` lub `average="weighted"` jeśli etykiety mają różną wagę).

---

### **Podrozdział: Klasyfikacja Wielowyjściowa (Multioutput Classification)**

*   **Streszczenie:** To uogólnienie klasyfikacji wieloelementowej. Tutaj każda etykieta może być wieloklasowa (a nie tylko binarna). Przykładem jest system do usuwania szumu z obrazków: na wejściu jest zaszumiony obrazek, a na wyjściu "czysty". Wyjściem jest wiele etykiet (jedna dla każdego piksela), a każda etykieta może przyjąć wiele wartości (0-255).

*   **Pojęcia i Listingi Kodu:**

    *   **Kod:**
        ```python
        # Dodajemy szum do obrazków
        noise = np.random.randint(0, 100, (len(X_train), 784))
        X_train_mod = X_train + noise
        # Etykietami są oryginalne, czyste obrazki
        y_train_mod = X_train

        knn_clf.fit(X_train_mod, y_train_mod)
        clean_digit = knn_clf.predict([X_test_mod[some_index]])
        ```
    *   **Teoria i Przeznaczenie:**
        *   Zadaniem modelu jest nauczenie się, jak odtworzyć oryginalny obrazek (`y_train_mod`) na podstawie jego zaszumionej wersji (`X_train_mod`).
        *   Jest to zadanie klasyfikacji wielowyjściowej, ponieważ model przewiduje 784 wartości (dla każdego piksela), a każda z tych wartości może być liczbą od 0 do 255.
        *   To pokazuje, jak elastyczne mogą być zadania uczenia maszynowego, zacierając granicę między klasyfikacją a regresją.

---

### **Schematyczny Plan Projektu Klasyfikacyjnego (Przepis)**

Oto uproszczona instrukcja krok po kroku, podsumowująca proces opisany w tekście:

1.  **Zdefiniowanie Problemu:**
    *   Określ, co chcesz przewidzieć. Czy to klasyfikacja binarna, wieloklasowa, wieloelementowa czy wielowyjściowa?

2.  **Pobranie i Wstępna Analiza Danych:**
    *   Załaduj dane (np. za pomocą `fetch_openml`).
    *   Sprawdź strukturę danych (`.keys()`, `.shape`).
    *   Zwizualizuj kilka przykładów, aby nabrać intuicji.

3.  **Przygotowanie Danych:**
    *   Wyczyść dane (np. zmień typy danych z tekstu na liczby).
    *   **Kluczowy krok:** Podziel dane na zbiór treningowy i testowy. Zbiór testowy odłóż i nie dotykaj go do samego końca.
    *   W razie potrzeby przeskaluj cechy (np. za pomocą `StandardScaler`).

4.  **Wybór i Trening Modelu:**
    *   Wybierz model początkowy (np. `SGDClassifier`, `RandomForestClassifier`).
    *   Wytrenuj model na zbiorze treningowym (`.fit()`).

5.  **Ocena Wydajności Modelu:**
    *   Użyj walidacji krzyżowej (`cross_val_score`, `cross_val_predict`) na zbiorze treningowym, aby uzyskać solidną ocenę.
    *   **Nie używaj dokładności (accuracy) dla niezbalansowanych zbiorów.**
    *   Przeanalizuj macierz pomyłek, aby zrozumieć rodzaje błędów.
    *   Oblicz precyzję, czułość i wynik F1.
    *   Narysuj krzywą Precision/Recall lub ROC, aby zwizualizować kompromisy i porównać modele.

6.  **Ulepszanie Modelu (Iteracja):**
    *   Na podstawie analizy błędów, spróbuj ulepszyć model:
        *   Zbierz więcej danych dla klas, z którymi model sobie nie radzi.
        *   Stwórz nowe cechy (inżynieria cech).
        *   Wypróbuj inne, bardziej zaawansowane modele.
        *   Dostrój hiperparametry modelu (np. próg decyzyjny).
    *   Powtarzaj kroki 4-6, aż uzyskasz satysfakcjonujący model.

7.  **Finalna Ocena:**
    *   Wytrenuj swój najlepszy, ostateczny model na **całym zbiorze treningowym**.
    *   Dokonaj **jednorazowej, ostatecznej oceny** na odłożonym wcześniej zbiorze testowym, aby uzyskać realistyczną miarę jego wydajności w "prawdziwym świecie".