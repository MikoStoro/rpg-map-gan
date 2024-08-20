# Tutorial do Google Colab
Colab znacznie przyspiesza trening, ale ma cały szereg ograniczeń, które znacznie spowalniają pracę i wprowadzają ryzyko błędów.

W Colabie możemy wygodnie korzystać jedybnie z plików, które umieściliśmy na Google Drive. Inne opcje to wgrywanie ich po każdej modyfikacji albo modyfikowanie ich w aplikacji webowej. Oba pomysły są tragiczne.

W związku z powyższym najefektywniejsza metoda pracy to edycja plików lokalnie, a następnie wgrywanie ich na drive przy pomocy narzędzia CLI (które jest szybsze i bardziej odporne na błędy niż przecąganie plikóœ myszką).

## Drive
Aby uzyskać dostęp do wgranych plików, należy wybrać opcję "Mount Google Drive" w menu plików.
Niestety nie można "zamontować" konkretnego folderu, ale to nie jest duży problem, ponieważ ścieżka do naszego folderu z GANem będzie zawsze jednakowa. 

## rclone - instalacja
Instrukcja instalacji rclone: 
https://rclone.org/install/
Pierwsze użycie po instalacji przeprowadzi cię przez wszystkie kroki konfiguracji (to niestety trochę trwa, ponieważ trzeba pozyskać Klucze API od Googla).

## rclone - komendy
Tutaj wszystko, czego możenie potrzebować

### rclone listremotes
wyświetla wszystkie dostępne remotes, czyli miejsca, z którymi możemy się synchronizować - Gdrive stanowi remote

### rclone tree <target:path>
ex. rclone tree gdrive_mikostoro:/
wyświetla pliki w danej lokacji

### rclone copy <source> <destination>
ex. rclone copy . gdrive_mikostoro:/gan_shared_folder
ex.  rclone copy models gdrive_mikostoro:/gan_shared_folder/models
Sugerowane do kopiowania plików na Dysk - nie na odwrót (chmura i git się nie lubią)
Można przesyłać pojedyncze pliki - przydatne przy wgrywaniu  datasetów (unikaj jednak uploadowania nieskompresowanych - taka operacja długo trwa)
uwaga: jeśli source jest folderem, to kopiowana jest jego zawartość - czyli w destination pojawi się szereg pojedynczych plików, nie cały folder
uwaga: przy zmianach w kodzie nie kopiuj całego repozytorium, a tylko folder z kodem, by zaoszczędzić czas

## Notatnik
W repozytorium znajduje się przykładowy notatnik. Będzie działał u was tylko, jeśli skopiujecie zawartość repozytorium do folderu /gan_shared_folder

## Ostrzeżenie
Aby uniknąć potencjalnych błędów i psucia repo:
- Nie edytuj w Colabie innych plików niż .ipynb
- Nie pobieraj zawartości folderu na Drive (szczególnie, jeśli znajdzie się tam plik .git)
