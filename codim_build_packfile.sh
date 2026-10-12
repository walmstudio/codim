#!/usr/bin/sh

clear

echo "Call of Duty 4: Modern Warfare PS3 Русификатор"
echo

# Ввод путей
read -r -p "Путь к packfile1.pak: " PAK_FILE
read -r -p "Путь к оригинальной папке voiceovers: " VO_ORIG
read -r -p "Путь к русской папке voiceovers: " VO_RUS
read -r -p "Путь к оригинальной папке battlechatter: " BC_ORIG
read -r -p "Путь к русской папке battlechatter: " BC_RUS

# Удаляем обрамляющие кавычки, если пользователь вставил их вместе с путём
for VAR in PAK_FILE VO_ORIG VO_RUS BC_ORIG BC_RUS; do
    VALUE="${!VAR}"
    VALUE="${VALUE#\"}"
    VALUE="${VALUE%\"}"
    printf -v "$VAR" '%s' "$VALUE"
done

# Проверка путей
if [[ ! -f "$PAK_FILE" ]]; then
    echo "Ошибка: файл packfile1.pak не найден: $PAK_FILE"
    exit 1
fi

if [[ ! -d "$VO_ORIG" ]]; then
    echo "Ошибка: оригинальная папка voiceovers не найдена: $VO_ORIG"
    exit 1
fi

if [[ ! -d "$VO_RUS" ]]; then
    echo "Ошибка: русская папка voiceovers не найдена: $VO_RUS"
    exit 1
fi

if [[ ! -d "$BC_ORIG" ]]; then
    echo "Ошибка: оригинальная папка battlechatter не найдена: $BC_ORIG"
    exit 1
fi

if [[ ! -d "$BC_RUS" ]]; then
    echo "Ошибка: русская папка battlechatter не найдена: $BC_RUS"
    exit 1
fi

# Проверка расположения инструментов
if [[ ! -f "codim_tools/codim_pak_tools_v3.py" ]]; then
    echo "Ошибка: запустите скрипт из папки, содержащей codim_tools/"
    exit 1
fi

# Фоновая музыка
env play --replay-gain track -q cod4_theme.mp3 repeat - \
    </dev/null >/dev/null 2>&1 &
PLAY_PID=$!

# Останавливаем музыку при завершении скрипта
trap 'kill "$PLAY_PID" 2>/dev/null || true' EXIT

echo
echo "Распаковка packfile1.pak"
python3 codim_tools/codim_pak_tools_v3.py extract "$PAK_FILE" paepak || exit 1

echo "Производится матчинг реплик"
python3 codim_tools/codim_matching_voices_v2.py \
    paepak \
    "$VO_ORIG" \
    voiceovers.json \
    --fast-accept 1.02 \
    --shortlist 24 \
    --verbose || exit 1

echo "Принудительный матчинг оставшихся реплик"
python3 codim_tools/codim_match_remaining.py \
    voiceovers.json \
    paepak/ \
    "$VO_ORIG" \
    voiceovers_v2.json || exit 1

echo "Проверка принудительных совпадений на адекватность длительности"
python3 codim_tools/codim_audit_forced.py \
    voiceovers_v2_forced.json \
    paepak/ \
    "$VO_ORIG" \
    --matches-json voiceovers_v2.json \
    --clean-output voiceovers_v2_clean.json || exit 1

echo "Применение ручных выборов voiceovers"
python3 codim_tools/codim_review_unmatched.py apply \
    voiceovers_v2_clean.json \
    review_choices.json \
    voiceovers_final.json || exit 1

echo "Карта звуков готова — запись русификатора voiceovers"
python3 codim_tools/codim_recode_matched.py \
    voiceovers_final.json \
    paepak/ \
    "$VO_RUS" \
    --tempo-steps 1.05,1.1,1.15,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.5 \
    --allow-trim || exit 1

echo "Матчинг battlechatter (треки voiceovers пропускаются)"
python3 codim_tools/codim_matching_battlechatter.py \
    paepak \
    "$BC_ORIG" \
    bc.json \
    --skip voiceovers_final.json \
    --verbose || exit 1

echo "Принудительный матчинг остатка battlechatter"

# Объединяем карты voiceovers и battlechatter,
# чтобы повторно не использовать уже занятые треки.
python3 - <<'EOF'
import json

with open("voiceovers_final.json", encoding="utf-8") as f:
    m = json.load(f)

with open("bc.json", encoding="utf-8") as f:
    m.update(json.load(f))

with open("bc_pool_guard.json", "w", encoding="utf-8") as f:
    json.dump(m, f, ensure_ascii=False, indent=2)
EOF

if [[ $? -ne 0 ]]; then
    echo "Ошибка при объединении карт звуков"
    exit 1
fi

python3 codim_tools/codim_match_remaining.py \
    bc_pool_guard.json \
    paepak/ \
    "$BC_ORIG" \
    bc_v2.json || exit 1

echo "Аудит длительности battlechatter"
python3 codim_tools/codim_audit_forced.py \
    bc_v2_forced.json \
    paepak/ \
    "$BC_ORIG" \
    --matches-json bc_v2.json \
    --clean-output bc_v2_clean.json || exit 1

echo "Ручные выборы battlechatter"
python3 codim_tools/codim_review_unmatched.py apply \
    bc_v2_clean.json \
    review_choices_bc.json \
    bc_final_all.json || exit 1

echo "Удаление voiceovers из карты battlechatter"
python3 - <<'EOF'
import json

with open("voiceovers_final.json", encoding="utf-8") as f:
    vo = json.load(f)

with open("bc_final_all.json", encoding="utf-8") as f:
    bc = json.load(f)

result = {
    track: record
    for track, record in bc.items()
    if track not in vo
}

with open("bc_final.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
EOF

if [[ $? -ne 0 ]]; then
    echo "Ошибка при подготовке итоговой карты battlechatter"
    exit 1
fi

echo "Запись battlechatter"
python3 codim_tools/codim_recode_matched.py \
    bc_final.json \
    paepak/ \
    "$BC_RUS" \
    --tempo-steps 1.05,1.1,1.15,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2,2.5 \
    --allow-trim || exit 1

echo "Упаковка модифицированного pak файла"
python3 codim_tools/codim_pak_tools_v3.py pack \
    paepak/ \
    packfile1_rus.pak || exit 1

echo
echo "Русификация готова!"
echo "Итоговый архив: packfile1_rus.pak"
