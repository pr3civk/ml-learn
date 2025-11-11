Jasne, przeanalizujmy dostarczony tekst i stwórzmy na jego podstawie kompleksowe, ale proste w odbiorze kompendium wiedzy na temat trenowania modeli, zakończone schematycznym planem działania.

### **Rozdział 4: Trenowanie Modeli – Streszczenie i Wyjaśnienie Kluczowych Pojęć**

Ten rozdział tłumaczy, co dzieje się "pod maską" modeli uczenia maszynowego. Zamiast traktować je jak czarne skrzynki, zagłębiamy się w mechanizmy ich działania, co pozwala lepiej dobierać modele, algorytmy i parametry, a także skuteczniej je debugować.

---

### **1. Regresja Liniowa (Linear Regression)**

**Teoria i Przeznaczenie:**
Regresja Liniowa to jeden z najprostszych modeli. Jej celem jest przewidzenie wartości liczbowej (np. ceny domu) na podstawie zestawu cech wejściowych (np. powierzchnia, liczba pokoi). Model uczy się, jak każda cecha wpływa na wynik, przypisując jej odpowiednią wagę. Przewidywanie to po prostu suma wszystkich cech pomnożonych przez ich wagi, plus stała wartość zwana "wyrazem wolnym" (bias/intercept).

Model stara się znaleźć taką linię (lub płaszczyznę w wielu wymiarach), która najlepiej pasuje do danych treningowych, czyli minimalizuje błąd między przewidywaniami a rzeczywistymi wartościami.

**Kluczowe Pojęcia i Implementacja:**

*   **Równanie Normalne (Normal Equation):**
    *   **Czym jest:** To matematyczna formuła, która pozwala obliczyć "jednym strzałem" idealne wartości wag dla modelu regresji liniowej, minimalizujące funkcję kosztu. Nie jest to proces iteracyjny (krok po kroku), ale bezpośrednie wyliczenie.
    *   **Przeznaczenie:** Używamy go, gdy chcemy znaleźć najlepsze parametry modelu szybko i bez potrzeby dostrajania hiperparametrów, takich jak współczynnik uczenia. Działa dobrze dla zbiorów danych, które nie są ekstremalnie duże pod względem liczby cech.
    *   **Dlaczego tak się robi:** To jak rozwiązanie równania matematycznego, które daje od razu poprawny wynik. Jest to wygodne, ale ma swoje wady. Główną wadą jest złożoność obliczeniowa – staje się bardzo powolne, gdy mamy tysiące cech, ponieważ wymaga obliczenia odwrotności dużej macierzy, co jest kosztowne obliczeniowo.

    *   **Przykład w kodzie:**
        ```python
        import numpy as np
        
        # Generowanie danych
        X = 2 * np.random.rand(100, 1)
        y = 4 + 3 * X + np.random.randn(100, 1)
        
        # Dodanie x0 = 1 do każdej instancji (dla wyrazu wolnego)
        X_b = np.c_[np.ones((100, 1)), X] 
        
        # Obliczenie wag za pomocą Równania Normalnego
        theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        ```
        *   `np.linalg.inv()`: Oblicza odwrotność macierzy. To serce Równania Normalnego.
        *   `.T`: Transpozycja macierzy (zamiana wierszy z kolumnami).
        *   `.dot()`: Mnożenie macierzy.

*   **Klasa `LinearRegression` z Scikit-Learn:**
    *   **Czym jest:** Gotowa do użycia klasa w bibliotece Scikit-Learn, która implementuje regresję liniową.
    *   **Przeznaczenie:** Zamiast ręcznie implementować Równanie Normalne, możemy użyć tej klasy. Jest zoptymalizowana i łatwa w użyciu. "Pod spodem" wykorzystuje bardziej zaawansowane i stabilne numerycznie metody (oparte na SVD - Singular Value Decomposition), które działają nawet w trudniejszych przypadkach.
    *   **Dlaczego tak się robi:** To standardowa praktyka. Rzadko kiedy pisze się takie algorytmy od zera, ponieważ gotowe implementacje są przetestowane, wydajne i obsługują wiele skrajnych przypadków.
    
    *   **Przykład w kodzie:**
        ```python
        from sklearn.linear_model import LinearRegression
        
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        
        # Dostęp do wyuczonych parametrów
        intercept = lin_reg.intercept_  # Wyraz wolny (theta_0)
        coefficients = lin_reg.coef_      # Wagi cech (theta_1, theta_2, ...)
        ```
        *   `lin_reg.fit(X, y)`: Ta metoda uruchamia proces "uczenia". Oblicza najlepsze wagi (`coef_`) i wyraz wolny (`intercept_`) na podstawie dostarczonych danych `X` i `y`.

---

### **2. Gradient Prosty (Gradient Descent)**

**Teoria i Przeznaczenie:**
Gradient Prosty to ogólny algorytm optymalizacyjny używany do znajdowania minimum funkcji – w naszym przypadku funkcji kosztu. Działa iteracyjnie:
1.  Zaczyna od losowych wartości wag.
2.  Oblicza gradient (nachylenie) funkcji kosztu – czyli kierunek, w którym funkcja rośnie najszybciej.
3.  Wykonuje mały krok w kierunku przeciwnym do gradientu (w dół "stoku").
4.  Powtarza kroki 2 i 3, aż dotrze do dna "doliny", czyli minimum funkcji kosztu.

**Dlaczego tak się robi:** W przeciwieństwie do Równania Normalnego, Gradient Prosty działa dobrze nawet z ogromną liczbą cech i może być używany do optymalizacji wielu innych modeli, nie tylko regresji liniowej.

**Kluczowe Pojęcia i Implementacja:**

*   **Współczynnik uczenia (Learning Rate):**
    *   **Czym jest:** Hiperparametr, który określa, jak duży krok robimy w każdej iteracji.
    *   **Przeznaczenie:** To kluczowy parametr do strojenia.
        *   **Zbyt mały:** Algorytm będzie zbiegał bardzo wolno, robiąc malutkie kroczki.
        *   **Zbyt duży:** Algorytm może "przeskoczyć" minimum i nigdy do niego nie trafić, a nawet zacząć się oddalać.
    *   **Wyzwanie:** Znalezienie "złotego środka" jest kluczowe dla efektywnego treningu.

*   **Skalowanie cech (Feature Scaling):**
    *   **Czym jest:** Proces przekształcania wartości wszystkich cech tak, aby miały podobny zakres (np. od 0 do 1 lub średnią 0 i odchylenie standardowe 1).
    *   **Przeznaczenie:** Jest to niezwykle ważne dla algorytmów opartych na gradiencie. Jeśli jedna cecha ma wartości w tysiącach (np. cena domu), a inna w jednostkach (np. liczba pokoi), funkcja kosztu będzie miała kształt bardzo wydłużonej, stromej "doliny". Algorytm będzie miał problem ze znalezieniem prostej drogi na dno i będzie zygzakował, co znacznie spowolni zbieżność.
    *   **Dlaczego tak się robi:** Skalowanie sprawia, że "dolina" funkcji kosztu staje się bardziej symetryczna, przypominając miskę. Dzięki temu algorytm może schodzić prosto w kierunku minimum, co znacznie przyspiesza trening. Używa się do tego np. klasy `StandardScaler` z Scikit-Learn.

**Warianty Gradientu Prostego:**

1.  **Wsadowy Gradient Prosty (Batch Gradient Descent):**
    *   **Jak działa:** W każdej iteracji oblicza gradient na podstawie **całego zbioru treningowego**.
    *   **Zalety:** Krok jest bardzo dokładny i stabilnie zmierza do minimum.
    *   **Wady:** Jest ekstremalnie wolny na dużych zbiorach danych, ponieważ każda aktualizacja wag wymaga przetworzenia wszystkich próbek.

2.  **Stochastyczny Gradient Prosty (Stochastic Gradient Descent - SGD):**
    *   **Jak działa:** W każdej iteracji losuje **jedną próbkę** ze zbioru treningowego i na jej podstawie oblicza gradient i aktualizuje wagi.
    *   **Zalety:** Bardzo szybki, ponieważ operuje na pojedynczych próbkach. Umożliwia trenowanie na ogromnych zbiorach danych, które nie mieszczą się w pamięci (out-of-core).
    *   **Wady:** Kroki są bardzo "hałaśliwe" i nieregularne. Algorytm nie schodzi gładko do minimum, ale "odbija się" wokół niego. Dzięki temu ma szansę ominąć lokalne minima, ale nigdy idealnie nie "osiada" w globalnym minimum.
    *   **Harmonogram uczenia (Learning Schedule):** Aby rozwiązać problem "odbijania się", stosuje się strategię stopniowego zmniejszania współczynnika uczenia w trakcie treningu. Na początku kroki są duże (by szybko zbliżyć się do celu), a z czasem stają się coraz mniejsze, co pozwala algorytmowi "uspokoić się" i precyzyjniej trafić w minimum.

    *   **Implementacja w Scikit-Learn:**
        ```python
        from sklearn.linear_model import SGDRegressor
        
        # max_iter - maksymalna liczba epok, tol - kryterium zatrzymania
        # penalty=None - brak regularyzacji, eta0 - początkowy współczynnik uczenia
        sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1)
        sgd_reg.fit(X, y.ravel()) # y.ravel() zmienia kształt wektora
        ```
        *   `SGDRegressor`: Klasa do regresji z użyciem SGD. Ma wiele hiperparametrów do kontroli procesu uczenia.

3.  **Mini-batch Gradient Descent:**
    *   **Jak działa:** Kompromis między dwoma powyższymi. W każdej iteracji oblicza gradient na podstawie **małej, losowej podgrupy (mini-batch)** danych.
    *   **Zalety:** Szybszy niż Batch GD. Mniej "hałaśliwy" niż SGD. Dodatkowo, wykorzystuje optymalizacje sprzętowe (np. na kartach GPU) do operacji na macierzach, co znacznie przyspiesza obliczenia.
    *   **Wady:** Może mieć trudniej z ominięciem lokalnych minimów niż SGD.
    *   **Dlaczego tak się robi:** Jest to najczęściej stosowany wariant w praktyce, zwłaszcza w głębokim uczeniu, ponieważ łączy zalety obu skrajnych podejść.

---

### **3. Regresja Wielomianowa (Polynomial Regression)**

**Teoria i Przeznaczenie:**
Co jeśli dane nie układają się w prostą linię, ale w krzywą (np. parabolę)? Możemy nadal używać modelu regresji liniowej! Trik polega na tym, żeby stworzyć nowe cechy, które są potęgami istniejących cech. Na przykład, jeśli mamy jedną cechę `x`, możemy dodać nową cechę `x^2`. Teraz model regresji liniowej będzie uczył się wag nie tylko dla `x`, ale też dla `x^2`, co pozwoli mu dopasować krzywą.

**Dlaczego tak się robi:** To sprytny sposób na modelowanie nieliniowych zależności za pomocą prostego i dobrze zrozumiałego modelu liniowego.

**Implementacja:**

*   **Klasa `PolynomialFeatures` z Scikit-Learn:**
    *   **Czym jest:** Transformator, który automatycznie generuje nowe cechy wielomianowe.
    *   **Przeznaczenie:** Bierzemy oryginalne dane i "przepuszczamy" je przez `PolynomialFeatures`, aby uzyskać rozszerzony zestaw danych z dodatkowymi cechami (np. `x^2`, `x^3`, a także kombinacjami, jak `a*b`, jeśli mamy cechy `a` i `b`).
    
    *   **Przykład w kodzie:**
        ```python
        from sklearn.preprocessing import PolynomialFeatures
        
        # degree=2 oznacza, że tworzymy cechy do potęgi drugiej
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X)
        
        # Teraz X_poly zawiera oryginalną kolumnę X oraz kolumnę X^2
        # Na tych danych można trenować zwykły model LinearRegression
        lin_reg = LinearRegression()
        lin_reg.fit(X_poly, y)
        ```
        *   **Uwaga:** Używanie wielomianów wysokiego stopnia prowadzi do **przeuczenia (overfitting)** – model idealnie dopasuje się do danych treningowych, ale będzie słabo generalizował na nowych danych.

---

### **4. Krzywe Uczenia (Learning Curves)**

**Teoria i Przeznaczenie:**
Krzywe uczenia to potężne narzędzie do diagnozowania, czy model jest zbyt prosty (**niedouczenie / underfitting**) czy zbyt złożony (**przeuczenie / overfitting**). Są to wykresy pokazujące błąd modelu na zbiorze treningowym i walidacyjnym w funkcji rozmiaru zbioru treningowego.

**Dlaczego tak się robi:** Zamiast zgadywać, czy model jest odpowiedni, możemy to zwizualizować.
*   **Niedouczenie (Underfitting):** Obie krzywe (treningowa i walidacyjna) szybko się stabilizują na wysokim poziomie błędu i są blisko siebie. Oznacza to, że model jest zbyt prosty, by nauczyć się wzorców w danych. **Dodawanie kolejnych danych nie pomoże.** Trzeba użyć bardziej złożonego modelu lub dodać lepsze cechy.
*   **Przeuczenie (Overfitting):** Błąd na zbiorze treningowym jest bardzo niski, ale błąd na zbiorze walidacyjnym jest znacznie wyższy. Jest między nimi duża przerwa. Oznacza to, że model "nauczył się na pamięć" danych treningowych, włączając w to szum, i nie potrafi generalizować. **Dodawanie kolejnych danych treningowych może pomóc** zmniejszyć tę przerwę. Innym rozwiązaniem jest uproszczenie modelu lub zastosowanie regularyzacji.

**Kluczowe Pojęcia:**
*   **Kompromis między obciążeniem a wariancją (Bias/Variance Trade-off):**
    *   **Obciążenie (Bias):** Błąd wynikający z błędnych założeń modelu (np. zakładanie liniowości, gdy dane są kwadratowe). Model o wysokim obciążeniu jest niedouczony.
    *   **Wariancja (Variance):** Błąd wynikający z nadmiernej czułości modelu na małe zmiany w danych treningowych. Model o wysokiej wariancji jest przeuczony.
    *   **Kompromis:** Zwiększenie złożoności modelu zazwyczaj zmniejsza obciążenie, ale zwiększa wariancję. Celem jest znalezienie złotego środka.

---

### **5. Regularyzowane Modele Liniowe**

**Teoria i Przeznaczenie:**
Regularyzacja to technika walki z przeuczeniem. Polega na "karaniu" modelu za posiadanie zbyt dużych wag. Do funkcji kosztu dodaje się dodatkowy składnik (tzw. "karę za regularyzację"), który rośnie wraz z wartościami wag. Algorytm optymalizacyjny stara się teraz zminimalizować nie tylko błąd dopasowania do danych, ale także tę karę. W efekcie wagi są "ściągane" w kierunku zera, co prowadzi do prostszego modelu.

**Dlaczego tak się robi:** To jak powiedzieć modelowi: "Staraj się jak najlepiej dopasować do danych, ale używaj jak najprostszych wyjaśnień (małych wag)". To zmusza go do generalizacji zamiast zapamiętywania.

**Typy Regularyzacji:**

1.  **Regresja grzbietowa (Ridge Regression):**
    *   **Jak działa:** Dodaje do funkcji kosztu karę równą sumie kwadratów wag (kara L2).
    *   **Efekt:** Zmniejsza wszystkie wagi, ale rzadko kiedy zeruje je całkowicie. Jest dobrym domyślnym wyborem.
    *   **Hiperparametr `alpha`:** Kontroluje siłę regularyzacji. `alpha = 0` to brak regularyzacji. Duże `alpha` sprawia, że wszystkie wagi dążą do zera.

2.  **Regresja Lasso (Lasso Regression):**
    *   **Jak działa:** Dodaje do funkcji kosztu karę równą sumie wartości bezwzględnych wag (kara L1).
    *   **Efekt:** Ma bardzo ciekawą właściwość: potrafi całkowicie wyzerować wagi najmniej istotnych cech. Działa więc jak **automatyczna selekcja cech**.
    *   **Kiedy używać:** Gdy podejrzewamy, że tylko niektóre cechy są naprawdę ważne.

3.  **Elastic Net:**
    *   **Jak działa:** Stanowi połączenie regularyzacji Ridge i Lasso. Posiada dodatkowy hiperparametr `r`, który kontroluje proporcje między karą L1 i L2.
    *   **Kiedy używać:** Jest często preferowany nad Lasso, ponieważ zachowuje się stabilniej, gdy cechy są silnie skorelowane lub gdy mamy więcej cech niż próbek.

4.  **Wczesne Zatrzymywanie (Early Stopping):**
    *   **Jak działa:** To zupełnie inny rodzaj regularyzacji, stosowany w algorytmach iteracyjnych (jak Gradient Prosty). Polega na monitorowaniu błędu na zbiorze walidacyjnym podczas treningu. Trening jest przerywany w momencie, gdy błąd na zbiorze walidacyjnym przestaje spadać i zaczyna rosnąć.
    *   **Dlaczego tak się robi:** To bardzo prosta i skuteczna technika. Zatrzymujemy model dokładnie w punkcie, w którym zaczyna się przeuczać.

---

### **6. Regresja Logistyczna (Logistic Regression)**

**Teoria i Przeznaczenie:**
Mimo słowa "regresja" w nazwie, jest to algorytm **klasyfikacji**. Służy do przewidywania prawdopodobieństwa, że dana próbka należy do określonej klasy (np. czy e-mail to spam, czy nie).
1.  Działa podobnie do regresji liniowej: oblicza ważoną sumę cech.
2.  Jednak zamiast zwracać tę sumę bezpośrednio, przepuszcza ją przez **funkcję logistyczną (sigmoidalną)**, która "ściska" wynik do zakresu od 0 do 1.
3.  Wynik ten jest interpretowany jako prawdopodobieństwo. Domyślnie, jeśli prawdopodobieństwo jest > 0.5, próbka jest klasyfikowana jako klasa pozytywna (1), w przeciwnym razie jako negatywna (0).

**Kluczowe Pojęcia:**
*   **Granica decyzyjna (Decision Boundary):** To linia lub płaszczyzna, która oddziela przewidywania dla różnych klas. W regresji logistycznej jest ona liniowa.

---

### **7. Regresja Softmax (Softmax Regression)**

**Teoria i Przeznaczenie:**
Jest to uogólnienie regresji logistycznej na problemy z **wieloma klasami** (np. klasyfikacja kwiatów na 3 gatunki: irys setosa, versicolor, virginica).
1.  Dla każdej klasy model oblicza osobny wynik (score).
2.  Następnie wyniki te są przepuszczane przez **funkcję softmax**, która przekształca je w wektor prawdopodobieństw, sumujących się do 1.
3.  Próbka jest przypisywana do klasy, która uzyskała najwyższe prawdopodobieństwo.

**Dlaczego tak się robi:** Pozwala to na bezpośrednie rozwiązanie problemu wieloklasowego w jednym modelu, zamiast trenować wiele osobnych klasyfikatorów binarnych (jak w strategii One-vs-Rest).

**Kluczowe Pojęcia:**
*   **Entropia krzyżowa (Cross-Entropy):** To funkcja kosztu używana w regresji Softmax. Mierzy, jak bardzo przewidywane prawdopodobieństwa różnią się od rzeczywistych etykiet. Celem treningu jest minimalizacja tej funkcji.

---

### **Schematyczny Plan Projektu (Przepis)**

Oto uproszczona, krok po kroku instrukcja tworzenia projektu uczenia maszynowego na podstawie wiedzy z tego rozdziału.

1.  **Przygotowanie Danych:**
    *   Załaduj dane.
    *   Oczyść dane (usuń braki, popraw błędy).

2.  **Podział Danych:**
    *   Podziel dane na zbiór treningowy i testowy (`train_test_split`). Zbiór testowy odłóż i nie dotykaj go aż do końcowej oceny modelu.

3.  **Eksploracja i Przetwarzanie Wstępne (na zbiorze treningowym):**
    *   Zbadaj dane – sprawdź skale cech, rozkłady.
    *   Jeśli to konieczne, stwórz nowe cechy (np. `PolynomialFeatures` dla nieliniowości).
    *   Zidentyfikuj potrzebę skalowania cech.

4.  **Budowa Potoku (Pipeline):**
    *   Stwórz potok (`Pipeline`) łączący kroki przetwarzania, np. skalowanie i model. To zapobiega wyciekowi informacji ze zbioru walidacyjnego/testowego do treningu.

5.  **Wybór i Trening Modelu:**
    *   **Problem liniowy, mało cech:** Zacznij od `LinearRegression` (Równanie Normalne).
    *   **Dużo cech:** Wybierz `SGDRegressor` lub inny model oparty na gradiencie. Pamiętaj o skalowaniu!
    *   **Podejrzenie przeuczania:** Wypróbuj modele z regularyzacją (`Ridge`, `Lasso`, `ElasticNet`). Dostrój hiperparametr `alpha`.
    *   **Problem nieliniowy:** Użyj `PolynomialFeatures` w potoku przed modelem liniowym.

6.  **Ocena i Diagnostyka Modelu:**
    *   Użyj metryk oceny (np. MSE, RMSE dla regresji) na zbiorze walidacyjnym (utworzonym z treningowego).
    *   Narysuj **krzywe uczenia**, aby zdiagnozować niedo- lub przeuczenie.
        *   **Niedouczenie?** -> Spróbuj bardziej złożonego modelu (np. wyższy stopień wielomianu) lub dodaj nowe, lepsze cechy.
        *   **Przeuczenie?** -> Zwiększ regularyzację (większe `alpha`), uprość model, zastosuj *Early Stopping* lub zdobądź więcej danych treningowych.

7.  **Strojenie Hiperparametrów:**
    *   Użyj technik takich jak Grid Search lub Randomized Search, aby znaleźć najlepsze wartości hiperparametrów (np. `alpha`, współczynnik uczenia).

8.  **Końcowa Ocena:**
    *   Po wybraniu i dostrojeniu finalnego modelu, przetestuj go **jeden raz** na odłożonym na początku zbiorze testowym, aby uzyskać ostateczną, bezstronną ocenę jego wydajności.