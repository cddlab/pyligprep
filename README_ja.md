# pyligprep

[English](README.md) | 日本語

リガンドPDBの水素付加、PySCF/GPU4PySCFでの構造最適化、HFのESP計算、内蔵した二段階RESP、AmberTools 26によるGAFF2パラメータ作成を行うPythonパッケージ。成果物は原子名を保持した `ligand.mol2`、`ligand.prep`（PREPI形式）、`ligand.frcmod`。`run` / `export` の `--resname SIN`（`-r SIN`）で、ファイル名を `SIN.*`、MOL2・PREP内の残基名を `SIN` に指定できる。大文字・小文字はそのまま保持する。

初版の対象は、単一分子・単一残基・単一配座の有機リガンド
（H、C、N、O、F、P、S、Cl）。金属、塩の混合物、複数モデル、未処理のaltlocは拒否する。
欠損した重原子の補完、pKa予測、配座探索、振動数による極小点確認は行わない。

## インストール

Python 3.12以降を必要とする。主な対象はGPUとSlurmを備えたLinux x86_64（Ubuntu 24.04で開発・検証）。
macOS（Apple Silicon）ではCPUのみで、小さな分子の構造最適化とパラメータ作成ができる（後述）。GPU extraはLinux x86_64/CUDA 12向け。

```bash
cd pyligprep
uv sync --extra cuda12
module load ambertools26 # AmberToolsのbinがPATHに入る
unset PYTHONPATH
source .venv/bin/activate
pyligprep doctor
```

CPUのみの場合は `uv sync`。既存のPython環境にローカルから導入する場合は `uv pip install '.[cuda12]'`（CPUのみなら `uv pip install .`）を使用する。
`hadd` がPDBのみから結合次数を推定するために使うOpen Babel wheel（`openbabel-wheel`）は標準でインストールされ、別プログラムの `obabel` として実行される。SMILES／テンプレートを指定する水素付加ではOpen Babelを使わない。

GPU4PySCF 1.9.1、CuPy CUDA12 13.4.1、cuTENSOR 2.2.0、GPU libxc 0.9.0をextraで指定している。開発環境の解決結果は `uv.lock` に保存。既存の手動ビルド環境には依存しない。AmberTools 26のバイナリ、NVIDIAドライバ、適合するCUDA実行環境は別途必要。この環境のAmberToolsモジュールはCUDA 12.8をロードするため、CUDA 12.9を併用しない。他の環境向けには同梱Slurmスクリプトのmodule設定を変更する。

### macOS（Apple Silicon、CPUのみ）

AmberTools 26はHomebrewの `brewsci/bio` tapから導入する。PySCFをOpenMP付きでビルドするため、
HomebrewのLLVM（`llvm`）も導入する（**推奨・既定**）。

```bash
brew install brewsci/bio/ambertools llvm
export AMBERHOME=/opt/homebrew/opt/ambertools
cd pyligprep
uv sync
source .venv/bin/activate
pyligprep doctor
```

`doctor` の `gpu.available` は `false`、`dispersion.d3_backend` は `simple-dftd3`、
`amber` に3つのコマンドが表示されれば準備完了。`pyligprep` はPATH上（Homebrewの `/opt/homebrew/bin`）の
`antechamber` 等を使う。`AMBERHOME` はAmberTools自身とテストが参照する。`cuda12` extraはLinux専用のため指定しない。

#### OpenMP付きPySCF（推奨）

PyPIのmacOS用PySCF wheelはOpenMPなしでビルドされており、`--threads` が効かない（`OpenMP is not available` の警告が出て1スレッドで動く）。そこで `pyproject.toml` の`[tool.uv.sources]` と `[tool.uv.extra-build-variables]` により、macOSでは `uv sync` 時にPySCFをソースからHomebrew LLVMのclangでビルドし、OpenMPを有効にしている（Linuxは従来どおりPyPIのwheel）。

- 初回の `uv sync` でPySCFのビルドに数分かかる。以後はuvのキャッシュを使う。
- Apple標準のclangはOpenMPに対応しておらず、警告なしにOpenMPなしでビルドされるため、LLVMのclangを指定している。
  パスはApple SiliconのHomebrew（`/opt/homebrew/opt/llvm`）を前提とする。`llvm` が未導入だとビルドに失敗する。
- PyPIのPySCF 2.15.0のsdistはビルドに必要な `libxcfun.patch` を含まず失敗するため、GitHubのリリースアーカイブ
  （v2.15.0）から取得している。macOSのPySCFはこのURLで2.15.0に固定されるため、更新時はURLを書き換える。
- この設定はuv専用（`uv sync` と `uv pip install` で有効）。uvを使わない `pip install` ではOpenMPなしのwheelが入る。
- OpenMPが有効かは次で確認できる（`8` が表示され、警告が出なければ有効）。

  ```bash
  python -c "from pyscf import lib; lib.num_threads(8); print(lib.num_threads())"
  ```

SINの構造最適化（12原子）では、wheel版98秒に対し `--threads 8` で44秒（出力ファイルは同一）。OpenMPを使わない場合は、`pyproject.toml` から上記2つのセクションを削除して `uv lock` し直す。

- D3(BJ)分散補正：Linuxは `pyscf-dispersion` を使うが、このパッケージにはmacOS用のwheelがない。
  macOSでは同じsimple-dftd3ライブラリのPython版 `dftd3` を自動で使う（`pyproject.toml` で
  プラットフォーム別に指定済み）。DFT-D4はmacOSでは使えない。
- `--device auto` はCPUへフォールバックする（警告が出る）。`--device cpu` を明示してもよい。
  `--device gpu` は停止する。
- 既定の `--memory-mb 22000` は実メモリより大きい場合がある。搭載メモリに合わせて `--memory-mb` を指定する。
- `--threads` は物理コア数（Apple Siliconでは高性能コア数）程度を目安にする。大きな分子はLinux/GPUで計算する。
- Slurmスクリプトは使わず、コマンドを直接実行する。
- 同じ分子でもLinuxで作ったステージをmacOSで `--resume` すると、依存バージョンが異なるため
  完了済みの結果は再利用されず再計算になる。

同梱のコハク酸ジアニオン（`examples/sin_h.pdb`、ほぼ最適化済みの座標）で全工程を実行する例：

```bash
pyligprep run -i examples/sin_h.pdb -o sin_mac \
  --charge -2 --multiplicity 1 --smiles 'O=C([O-])CCC(=O)[O-]' \
  --device cpu --threads 8 --memory-mb 8000 --resname SIN
```

`sin_mac/SIN.{mol2,prep,frcmod}` と `sin_mac/versions.json` が出力され、作業ディレクトリ `sin_mac/tmp/` は成功後に削除される。
`examples/` にはサイドカーJSONを同梱していないため、`hadd` 時と同じ `--smiles` を指定する。

## CLI一覧と入力・出力の指定

以下のコマンド例は仮想環境を有効にした状態で実行する。
有効にしない場合は `pyligprep` を `.venv/bin/pyligprep` に置き換える。同梱Slurmスクリプトは、`pyproject.toml` のある `pyligprep/` を作業ディレクトリとして投入する。例の電荷−1・一重項は対象リガンドに合わせて変更する。

| サブコマンド | 用途 | 主な入力 → 出力 |
| --- | --- | --- |
| `doctor` | 依存パッケージ、GPU、AmberToolsの検出 | 実行環境 → 標準出力のJSON |
| `hadd` | 水素付加と確認用PDBの作成 | PDB → 水素付きPDB＋サイドカーJSON |
| `check` | 原子名・結合・電荷・多重度の整合性確認 | 水素付きPDB → 検証結果 |
| `opt` | 構造最適化（工程A） | 水素付きPDB → 最適化ステージ |
| `resp` | HF/ESP計算と二段階RESP（工程B） | PDBまたは完了済みoptステージ → RESPステージ |
| `export` | GAFF2割り当てと電荷の丸め・総和補正（工程C） | 完了済みRESPステージ → MOL2・PREP・FRCMOD |
| `run` | A → B → Cを連続実行 | 確認済みの水素付きPDB → 3工程の結果 |

環境確認や各コマンドの引数は次のように確認できる。

```bash
pyligprep --version   # -V でも同じ。例: pyligprep 0.0.1
pyligprep --help
pyligprep opt --help
pyligprep resp --help
pyligprep export --help
pyligprep doctor --amber-bin /home/apps/ambertools26/bin
```

すべてのサブコマンドで `-V` / `--version` が使え、`--help` の最初の行にも `pyligprep 0.0.1` のようにバージョンが表示される。
この番号は `pyproject.toml` の `project.version` で、インストール済みパッケージのメタデータ経由で読み込む。
バージョンを変えたら `uv sync` で再インストールすると表示に反映される。

`doctor` は現在の実行ノードを調べる。GPUが割り当てられていないログインノードでは
GPU利用不可と表示されることがある。`doctor` と `check` は構造最適化を開始しない。

| 引数 | 意味と使用条件 |
| --- | --- |
| `-i FILE` / `--input FILE` | 入力PDB。`hadd`、`check`、`opt`、`resp`、`run`で使用 |
| `-o PATH` / `--output PATH` | `hadd`では出力PDB名。`opt`、`resp`、`export`、`run`では**新規の出力ディレクトリ** |
| `--charge INTEGER` | 分子全体の整数電荷。PDBから`check`、`opt`、`resp`、`run`を始める場合に必須 |
| `--multiplicity INTEGER` | スピン多重度2S+1。一重項は1、二重項は2。上記と同じ条件で必須 |
| `--from DIRECTORY` | `resp`では完了済みopt、`export`では完了済みRESPのステージディレクトリ |
| `--resume DIRECTORY` | 同じサブコマンドの出力ディレクトリを、その `tmp/` 内の作業ファイルから再開 |
| `--keep-tmp` | `run` のみ。成功後も `tmp/` を残す（既定では削除）。`run` が途中で止まった場合は常に残る |
| `--smiles STRING` / `--template FILE` | PDBからの結合次数・形式電荷の割り当て。どちらか一方を指定 |
| `--amber-bin DIRECTORY` | AmberToolsのbinディレクトリ。`doctor`、`export`、`run`で使用。未指定時はPATHから探す |
| `-r NAME` / `--resname NAME` | `run`、`export`の出力ファイル名のprefixと残基名。大小文字を保持。先頭が英字の1～3文字のASCII英数字 |

`-i`、`--from`、`--resume` は同時に指定しない。新規計算では `-o` が必要で、再開時には `--resume` が出力先も指定するため `-o` は不要。
`--from` / `--resume` に渡すのは前の工程の出力ディレクトリ（`-o`）であり、XYZやチェックポイント単体のパスではない。旧バージョンが書いたディレクトリ（`manifest.json` / `attempts/` を持つもの）は再開にも `--from` にも使えないため、新しい出力先で実行し直す。保存済みの総電荷・多重度と異なる値を後から指定することはできない。電子状態を変える場合はPDBから新しい出力先へ開始する。

## 1. 水素を付加して確認する

```bash
pyligprep hadd -i ligand.pdb -o ligand_h.pdb
```

出力は `ligand_h.pdb` と `ligand_h.pyligprep.json`。元の原子の名前・順番・座標を保持し、新しい水素は重複しない名前で末尾に追加する。既存の水素は削除しない。JSONには結合次数、形式電荷、SMILES、追加した水素の結合先を保存する。

**ユーザーが構造とプロトン化状態を確認してから、optまたはrunを実行する。**

PDBだけからの結合次数・プロトン化状態は推定である。目的の状態を指定するには、
例えば次のようにSMILESまたは単一分子のSDF/MOL/MOL2テンプレートを使用する。

```bash
# このSMILESは酢酸イオンの例。対象リガンドのSMILESに置き換える。
pyligprep hadd -i acetate.pdb -o acetate_h.pdb \
  --smiles 'CC(=O)[O-]' --charge -1

pyligprep hadd -i ligand.pdb -o ligand_h.pdb \
  --template ligand.sdf --charge -1
```

`hadd --charge` は検証条件であり、指定した総電荷に合わせてプロトン化部位を選ぶ機能ではない。予測した電荷と異なる場合は停止する。`run` が水素を自動追加することもない。入力PDBの原子名は残基内で一意でなければならない。また、原子名の先頭は数字であってはならない（GaussViewなどが出力する `1H1`・`2H10` など）。antechamberがMOL2作成時に元素を推定できないためで、`hadd` / `check` / `opt` / `run` はこうした入力を拒否する。あらかじめ `H1A` などに名前を変えておくこと。

PDB編集後はサイドカーのハッシュが一致しなくなる。構造を修正した場合は、修正済みPDBを別名の入力にして、適切なテンプレートで `hadd` を再実行する。既存水素の除去が必要なプロトン化変更は、入力を修正してから行う。

開発時に扱った実リガンド（ligcl3）では、PDBのみから推定した結果がC30H48O3、電荷0となり、目的の状態（C30H45O3、電荷−1）とは異なった。特定のプロトン化状態が目的なら、対応する化学構造テンプレートを必ず与える。詳細は [検証記録](docs/validation.md) を参照。

## 2. 計算前の確認

```bash
pyligprep check -i ligand_h.pdb --charge -1 --multiplicity 1
```

入力PDBから `check`、`opt`、`resp`、`run` を開始するときは、総電荷と多重度が必須。`--from` と `--resume` では、検証済みの保存データから引き継ぐ。
既に水素が揃っているPDBは `hadd` を省略できる。結合推定が失敗する場合は、`--smiles` または `--template` を指定する。

## 3. 個別の工程

### A. 構造最適化：`opt`

```bash
pyligprep opt -i ligand_h.pdb -o work/opt \
  --charge -1 --multiplicity 1 --device gpu
```

既定はBerny、B3LYP-D3BJ/6-31+G(d,p)。アルゴリズムや計算条件を変更する例：

```bash
pyligprep opt -i ligand_h.pdb -o work_geometric/opt \
  --charge -1 --multiplicity 1 --device gpu \
  --optimizer geometric --method B3LYP-D3BJ --basis '6-31+G(d,p)' \
  --opt-maxcycles 400 --threads 8 --memory-mb 22000
```

`--optimizer berny` でBernyを明示できる。括弧を含む基底関数名は上の例のように引用符で囲む。`--method` と `--basis` は最適化の設定であり、後続のESP計算のHF/6-31G*には影響しない。最適化の最終構造は `work/opt/optimized.xyz` と `work/opt/optimized.pdb` に出力される。後続工程には `--from work/opt` を使用する。

### B. ESPと二段階RESP：`resp`

```bash
pyligprep resp --from work/opt -o work/resp --device gpu
```

最適化ステージから原子名・結合・全桁の座標・総電荷・多重度を引き継ぐ。
使用装置やスレッド数はこのコマンドでも指定できる。
RESPの反復上限や確認済みの等価原子拘束を指定する場合：

```bash
pyligprep resp --from work/opt -o resp_custom --device gpu \
  --max-iterations 100 --resp-constraints reviewed_constraints.json
```

`reviewed_constraints.json` は原子順に対応した `stage1` / `stage2` 配列を持つ
既存ファイルを指定する。形式は後述の「再開と保存データ」を参照。
ESPとRESPは一つのサブコマンドで実行し、各計算結果は `tmp/` に別々に保存する。
出力は `work/resp/charges.json`（原子名、最終的なRESP電荷、各段階の反復回数とESPのRMS誤差）。

### C. MOL2・PREP・FRCMODの作成：`export`

```bash
pyligprep export --from work/resp -o parameters

# PATHにAmberToolsのコマンドがない場合
pyligprep export --from work/resp -o parameters_explicit \
  --amber-bin /home/apps/ambertools26/bin
```

`export` は `antechamber`、`parmchk2 -s gaff2`、`tleap` を使用する。
PATHにない場合は `--amber-bin /path/to/ambertools/bin` を指定する。
`parameters/ligand.mol2`、`parameters/ligand.prep`、`parameters/ligand.frcmod` に出力される。
MOL2の原子名・順番・電荷・結合と、PREPIの実原子の名前・電荷・結合を検証する。
PREPIにはダミー原子と形式上の並び替えがある。両形式についてLEaPでトポロジーを作成し、
未解決パラメータやエラーがあれば成功扱いしない。

ファイル名と残基名を指定する場合は、次のように `--resname` または `-r` を付ける。

```bash
pyligprep export --from work/resp -o sin_parameters --resname SIN
```

この例では `sin_parameters/SIN.mol2`、`SIN.prep`、`SIN.frcmod` を作成する。
MOL2のMOLECULE名、ATOM行の残基名、SUBSTRUCTURE名と、PREPの `SIN INT 0` の
残基名を一致させる。FRCMODには残基名の欄がないため、ファイル名へ反映する。
`--resname SiN` ならファイル名・残基名とも `SiN` になる。
未指定時はファイル名を `ligand.*`、残基名を入力PDB由来の値とする。
原子名、座標、RESP計算結果は変更しない。

出力電荷は小数点以下5桁で四捨五入し、6桁目を0として書く（例：`0.047200`）。
ちょうど中間の場合は絶対値を大きくする方向に丸める。丸めによる総和のずれは
`0.000010 e` 単位の整数計算で補正し、ファイルの電荷総和を指定整数に厳密に一致させる。
0・−1・+1に近い電荷の組を優先し、RESPの第2段階の等価拘束と、第2段階で
固定された原子間の第1段階の等価拘束を維持する。同じ数値だったという理由だけで
無関係な原子を等価とはみなさない。

まず各組の電荷を1単位だけ増減する組み合わせを探し、残差を合わせられない組は
飛ばす。必要なら複数組を逆方向に動かすか複数単位補正し、変更量を抑えて整合させる。
例外として、全原子が3個ずつの等価組に属している電荷+1の分子などでは、5桁精度と
等価性を同時に守った整数総和は数学的に不可能である。この場合は等価性を崩さずに
明示的なエラーとする。

補正前のRESP電荷、等価組、丸め値、補正単位数は `tmp/charge-rounding.json` に記録する。
元のRESP計算結果は変更しない。`.prep` の内部形式およびantechamberの
`-fi` / `-fo` の指定は引き続き `prepi`。MOL2とPREPの実原子の電荷は同一である。

電荷の5桁化・総和補正は `export` / `run` で自動適用するため、追加フラグは不要。
`-o` は出力ディレクトリを指定する引数であり、`-o ligand.mol2` のように
ファイル名を指定する引数ではない。どのコマンドも、その直下には成果物と `versions.json` だけを書き、
それ以外のファイルはすべて中の `tmp/` に置く。

```
parameters/                 # export（または run）の -o
├── ligand.mol2             # 成果物
├── ligand.prep
├── ligand.frcmod
├── versions.json           # 使用した pyligprep・Python・Pythonパッケージ・AmberToolsバイナリの版
└── tmp/                    # 作業ファイル、ログ、--resume / --from が使う状態
```

| コマンド | `-o` 直下の成果物 |
| --- | --- |
| `opt` | `optimized.pdb`、`optimized.xyz` |
| `resp` | `charges.json` |
| `export` / `run` | `ligand.mol2`、`ligand.prep`、`ligand.frcmod`（`--resname` 指定時は `ligand` がその値になる） |

| ファイル | 内容 |
| --- | --- |
| `ligand.mol2` | 入力PDBの原子名・順番、最適化後座標、GAFF2タイプ、補正後のRESP電荷 |
| `ligand.prep` | MOL2から変換したPREPI形式。実原子の名前・電荷はMOL2と同じ |
| `ligand.frcmod` | `parmchk2 -s gaff2` が出力する補足パラメータ |
| `versions.json` | 現在の成果物のファイル名（以前の `--resname` のファイルが並んで残っている場合がある）と、pyligprep・Python・Pythonパッケージの版。`export` / `run` では `antechamber` / `parmchk2` / `tleap` のパス・SHA-256・AmberToolsのリリース名も含む |
| `tmp/charge-rounding.json` | 元のRESP電荷、等価組、丸め・総和補正の記録 |
| `tmp/atom-map.json` | 出力MOL2の原子番号・名前・タイプ・電荷の対応 |
| `tmp/state.json` | その工程の入力・設定・依存バージョン・状態・結果。`--resume` / `--from` が使う |

### 構造最適化済みの入力から開始する

既に別のプログラムで最適化済みなら、PDBからBへ直接進める。

```bash
pyligprep resp -i optimized.pdb -o work/resp \
  --charge -1 --multiplicity 1 --device gpu
```

この入口は渡された構造に対する単点計算であり、最適化済みであることを自動判定しない。
`resp --from work/opt` は最適化の収束を検証する。
工程間ではPDBの丸め前の座標を使用するため、原則として `--from` を使用する。
`-i` はPDB入力であり、任意のXYZファイルを直接入力する機能や、PDBの原子名と
別ファイルのXYZ座標を自動結合するCLIは現時点では用意していない。

保存済みRESPから出力だけを作り直す場合も、HF/ESP/RESPは再計算しない。

```bash
# 既存exportを再開し、必要な場合だけ再出力
pyligprep export --resume parameters

# 別の新規ディレクトリへ出力する場合
pyligprep export --from work/resp -o parameters_new
```

CLIが指定先ディレクトリ直下に書き出すのは `ligand.mol2`、`ligand.prep`、`ligand.frcmod`
（`--resname` 指定時はその名前）と `versions.json` だけである。ほかの場所へコピーしたファイルは更新されない。

## 4. 一括実行とSlurm

```bash
pyligprep run -i ligand_h.pdb -o work \
  --charge -1 --multiplicity 1 --device gpu
```

最適化・RESP・出力を、`work/tmp/opt`、`work/tmp/resp`、`work/tmp/export` で順に実行する。
成功後に残るのは `work/ligand.{mol2,prep,frcmod}` と `work/versions.json` だけで、`work/tmp/` は削除される。
ログを確認したい場合や、あとで `run --resume` により設定を変えて再実行したい場合は、`--keep-tmp` を付けて `work/tmp/` を残す。
`run` が途中で止まった場合は `work/tmp/` が常に残り、`run --resume work` で止まった工程から再開できる。
`run` にも `--resname` / `-r` を指定できる。例えば次のコマンドでは
`sin_named_run/SIN.{mol2,prep,frcmod}` に出力する。

```bash
pyligprep run -i sin_h.pdb -o sin_named_run \
  --charge -2 --multiplicity 1 --device gpu --resname SIN
```

各サブコマンドと同じPython関数を使い、PyRESPの外部Pythonスクリプトは起動しない。
`hadd` は一括実行に含まれない。`run` にはユーザーが確認した水素付きPDBを渡す。
最適化用の `--optimizer` / `--method` / `--basis` と、RESP用の
`--max-iterations` / `--resp-constraints` などを同時に指定できる。

同梱のSlurmスクリプトは開発に使った計算機（partition `q4`、`/home/apps` 以下のmodule）向けの例である。
他の環境ではpartition、module名、初期化パス、仮想環境のパスを書き換える。

| スクリプト | 用途 | 要求資源 |
| --- | --- | --- |
| `scripts/run_sin_slurm.sh` | コハク酸ジアニオン（SIN）の `check` と `run` を行う例 | 8タスク×1 CPU、1 CPUあたり7 GB、GPU指定なし、時間指定なし |
| `scripts/test_gpu_slurm.sh` | `gpu` マーカーのテスト（AmberTools・参照テストも有効） | GPU 1基、CPU 4、メモリ24 GB、上限1時間 |

`run_sin_slurm.sh` は、環境変数 `project` が指すディレクトリの `sin_h.pdb` と `sin_resp_constraints.json` を読み込み、
`$project/sin_run` に出力する。仮想環境のパス（`/home/moriwaki/...`）と `--amber-bin` はスクリプト内に固定されている。
`run` には `--device gpu` を指定しているが、スクリプト自体はGPUを要求していない。GPUのないノードに割り当てられると停止するため、
GPUを要求するオプションを `sbatch` に付けて投入する。

```bash
project=$PWD/sin_work sbatch --gpus=1 scripts/run_sin_slurm.sh

# 途中で止まった計算を再開する（--resume 以降は pyligprep run に渡される）
project=$PWD/sin_work sbatch --gpus=1 scripts/run_sin_slurm.sh --resume --opt-maxcycles 400
```

Slurmの標準出力・標準エラーは `run_sin_slurm.sh.JOBID.out` / `.err`。
`test_gpu_slurm.sh` はパッケージのディレクトリから `sbatch scripts/test_gpu_slurm.sh` で投入し、一時ファイルと
カバレッジを `validation/` に書く。`--cov` を使うため、`pytest-cov` を仮想環境に追加しておく（dev依存には含まれない）。

同じ出力先に対して複数のジョブを同時に投入しない。
`--threads` や `--memory-mb` はアプリケーション側の設定であり、Slurmの割り当て量は
変更しない。割り当て量も変える場合は `sbatch` のオプションをスクリプト名の前に指定する。

## 計算条件

| 項目 | デフォルト／指定方法 |
| --- | --- |
| 最適化 | Berny、B3LYP-D3BJ/6-31+G(d,p) |
| アルゴリズム | `--optimizer berny` または `--optimizer geometric` |
| DFT | `--method`、`--basis` |
| スピン | 閉殻一重項RKS、それ以外UKS。`--scf-type rks` / `uks` で明示可（`opt` / `run` のみ） |
| SCF | DFTは密度フィッティングとNewton SCF（最大100サイクル）、積分グリッドlevel 3 |
| 収束 | SCFエネルギー1e-10・勾配1e-6。Bernyは gradientmax 1.5e-5、gradientrms 1e-5、stepmax 6e-5、steprms 4e-5（原子単位） |
| 最適化の反復回数上限 | `--opt-maxcycles 200`（Gaussianの `Opt=MaxCycles` に対応） |
| 初期信頼半径 | `--opt-maxstep N`：Gaussianの `Opt=MaxStep=N` と同じ入力スケールの正整数。既定では未指定 |
| geomeTRIC | GAU_TIGHT、初期解析Hessian、trust=0.05、tmax=0.10 |
| 幾何拘束 | geomeTRICのみ `--constraints constraints.txt`、原子番号は1始まり |
| 計算装置 | `--device auto` / `gpu` / `cpu` |
| 資源 | `--threads 8 --memory-mb 22000` |
| SCFの反復回数上限 | `--scf-maxcycle 512`（Gaussianの `SCF=MaxCycle` に対応。DFTのNewton SCFでは100が上限） |
| ESP | 気相HF/6-31G*、Cartesian d、閉殻RHF／開殻UHF、密度フィッティングなし |
| ESP点群 | GPU4PySCF 1.9.1のGAMESS半径、倍率1.4/1.6/1.8/2.0、`--density 1.0` 点/Å² |
| RESP | 拘束強度 `--qwt1 0.0005 --qwt2 0.001`、双曲線半径0.1、水素は拘束対象外 |
| RESP収束 | `--tolerance 1e-6 --max-iterations 42`、PyRESPと同じ判定式 |
| RESP等価性 | `--equivalence-depth 3`（1～5）、明示指定は `--resp-constraints FILE` |

最適化用の引数は `opt` / `run`、ESP・RESP用の引数は `resp` / `run` に指定する。
`--device` / `--threads` / `--memory-mb` / `--scf-maxcycle` は
`opt` / `resp` / `run` に共通。`export` は保存済み電荷からファイルを生成するため
GPUを必要とせず、これらの量子計算用引数は受け取らない。
`--scf-maxcycle` はESP用のHFにはそのまま適用されるが、DFTのNewton SCFでは100が上限になる。
ESP用のHFは `--scf-type` に関係なく、閉殻一重項ならRHF、それ以外はUHFになる。

Bernyの内部座標閾値とgeomeTRICのCartesian座標閾値は同じものではない。

`--opt-maxstep N` は**構造最適化の初期信頼半径**を指定する。Gaussian 16の `Opt=MaxStep=N` に合わせ、
Bernyの内部座標では `0.01 * N` Bohr/radianとする。たとえば `--opt-maxstep 10` は初期半径 `0.1`、
`--opt-maxstep 30` は `0.3`。指定できる値は正整数のみ。
geomeTRICでは長さをÅに換算し、`--opt-maxstep 10` は約 `0.052918 Å` となる。
geomeTRICはCartesian変位に基づくため、この換算で両最適化法やステップがGaussianと同等になるわけではない。
`tmax` は通常 `0.10 Å` を維持し、指定した初期半径がそれを超える場合は、暗黙の切詰めを避けるため
初期半径と同じ値まで引き上げる。

新規計算で省略した場合は従来の既定値を維持する：Bernyは `0.3`、geomeTRICは `0.05 Å`、`tmax=0.10 Å`。
再開時に省略すると保存済みの値を引き継ぎ、明示すると上書きする。保存設定にこの項目がない場合はバックエンドの既定値を使う。
最適化中はアルゴリズムが半径を調整する。このオプションは固定の最大ステップ、構造収束の変位閾値
（`stepmax` / `steprms`）、反復回数（`--opt-maxcycles`）、SCFの軌道回転ステップを指定するものではない。

`auto` はGPU初期化が利用できない場合にCPUへフォールバックする。
GPUで始めた後のSCF未収束・メモリ不足などをCPU再実行で隠すことはない。
`gpu` 指定時はGPUがなければ停止する。

開殻HFではα＋β密度からESPを作る。通常の閉殻AMBER/RESP手順と異なる電子状態には、
電荷・力場の適用妥当性を別途確認する必要がある。

## 再開と保存データ

再開時は元の入力・設定を読み込み、追加で指定した設定だけを変更する。
新規実行と同じ `-i ... -o ...` を既存ディレクトリに対して繰り返すと、上書きせず停止する。

Pythonのオプションフィールドと保存設定のキーは `opt_max_cycles`、`opt_max_step`、`scf_max_cycle`。
旧引数 `--max-steps`、`--opt-trust`、`--max-scf-cycles` は廃止し、互換エイリアスは設けない。
旧キー `max_steps`、`opt_trust`、`max_scf_cycles` を含む保存状態からは再開できないため、
新しい出力ディレクトリを指定して新規計算する。既存の計算ファイルは自動移行しない。

```bash
# ステップ不足で失敗した最適化を、保存済み構造から続行
pyligprep opt --resume work/opt --opt-maxcycles 400

# 同じ構造から最適化アルゴリズムを切り替え
pyligprep opt --resume work/opt --optimizer geometric

# 初期信頼半径を指定して保存構造から再開（GaussianのMaxStep=10と同じ入力スケール）
pyligprep opt --resume work/opt --opt-maxstep 10

# RESPフィットの反復不足なら、完了済みESPを再利用して続行
pyligprep resp --resume work/resp --max-iterations 100

# 出力だけを再開／更新
pyligprep export --resume parameters

# 保存済みRESPを使い、ファイル名・残基名を変更して出力する
pyligprep export --resume parameters -r SIN

# 一括実行はその出力ディレクトリを指定する（失敗後、または --keep-tmp 使用時のように tmp/ が残っている必要がある）
pyligprep run --resume work
```

同じサブコマンドのステージを指定する。たとえば `opt --resume work` や `resp --resume work/opt` ではなく、`opt --resume work/opt`、`resp --resume work/resp` とする。GPUノードで再開する場合は、上記のコマンドを書いたジョブスクリプトを `sbatch` で投入する（同梱の `run_sin_slurm.sh` はSIN専用の例）。

RESPをやり直して電荷が変わった場合、既存の `export --resume parameters` はそのexportに保存した以前のRESPを使う。新しいRESPを反映するには`export --from work/resp -o parameters_updated` のように新規出力先を指定する。`run --resume work` は `work/tmp/` 内の子ステージを順に確認し、最新のRESP結果をexportへ渡す。

`--resume` で `--resname` を省略すると、そのステージに保存済みの指定を引き継ぐ。明示すると新しい名前へ変更する。`run --resume work --resname SIN`（`--keep-tmp` で `tmp/` を残した場合）でも、計算条件が同じで完了済みの最適化・ESP・RESPは再利用し、出力工程だけを更新する。以前の名前のファイルは残す。手編集した出力は（`run` の出力ディレクトリ直下のコピーも含め）上書きせず停止するため、同じ名前で作り直す場合は `export --from ... -o 新規ディレクトリ --resname SIN` を使う。

- `tmp/state.json` に入力、設定、依存バージョン、状態、完了した各段階の記録を保存する。
- 試行履歴は持たない。再開は同じ `tmp/` で続きから行い、ログは上書きされる。
- 最適化は `last.xyz`、`last.json`、SCFチェックポイント、軌跡（再開をまたいで追記）を保存する。
- RKS／UKSチェックポイントの形式を検出し、密度を変換してSCFを再収束させる。
  UKS→RKSではスピン分極を破棄する。最適化器の内部履歴は復元しない。
- Method／Basisを変えた場合は、最新座標を使ってSCFを新規に行う。
- HF、ESP、第1段階RESP、第2段階RESPを別々に保存し、再開時は最後に完了した段階の続きから計算する。
  フィットだけの変更にはESPを再利用する。
- 出力を小さく保つため、SCFチェックポイント（`*.chk`）は工程の完了時に削除する。そのため、RESP工程の完了後に
  ESP点群を変更するとHFから再計算になる。
- 構造が変わった場合、`resp` / `export` は自身の `tmp/` を消去してから、その構造で再計算する。
- 完了した結果はハッシュを検証して再利用する。途中構造を収束済みと扱わない。
- 同一工程の同時書き込みをロックで拒否し、ユーザーが編集した最終出力を上書きしない。

`resp` は `tmp/` に `constraints.json`、`esp.npz`（Bohr／原子単位）、`esp.dat`、`stage1.in`、`stage2.in`、各段階の電荷JSONを保存する。`.in` は確認・比較用であり実行しない。等価原子の自動判定はPyRESP_GENを基準に、元素・混成・結合先の整合を確認する方式。大きな分子や特殊な官能基では制約を確認すること。
`--resp-constraints constraints.json` で `stage1`／`stage2` 配列を指定できる。
各要素は `0=自由`、`-1=固定`、`正の整数=同じ電荷にする先行原子の1始まり番号`。
配列の順番は入力PDBと同じ。総電荷制約は常に適用する。

## 開発・検証・配布

```bash
uv sync --extra cuda12
uv run --no-sync ruff check src tests
uv run --no-sync ruff format --check src tests
uv run --no-sync ty check src tests
uv run --no-sync pytest -q
uv build
```

macOSでは `uv sync` とする（`cuda12` はLinux専用。PySCFはOpenMP付きでソースからビルドされる）。
通常のテストはGPU、AmberTools、元のPyRESPを必要とするテストをスキップする。CPUのHF／ESPとD3(BJ)分散補正の値・勾配はテストするが、CPUでの構造最適化は実走しない。
macOSではHomebrewのAmberToolsで `PYLIGPREP_TEST_AMBER=1` / `PYLIGPREP_TEST_REFERENCE=1` のテストも実行できる。

テストは AmberTools を `PYLIGPREP_AMBER_BIN=/path/to/ambertools/bin`、`$AMBERHOME/bin`、PATH の順に探す。
これらの環境変数はテスト専用で、CLIは `--amber-bin` か PATH 上のコマンドだけを使う。

実施結果は [docs/validation.md](docs/validation.md) を参照。wheelとsdistは `dist/` に生成する。

Copyright (c) 2026, Yoshitaka Moriwaki, CDD Laboratory.
ライセンスはGNU General Public License バージョン3のみ（GPL-3.0-only、[LICENSE](LICENSE)）。RESPはAmberTools 26のPyRESP／PyRESP_GEN（GPL-3.0-only）、
ESP点群とGAMESS半径はGPU4PySCF 1.9.1（Apache-2.0）から移植しており、出典を
`src/pyligprep/resp.py` と `src/pyligprep/esp.py` の冒頭に記載している。AmberToolsのバイナリとデータベースは同梱しない。
Open Babel（`openbabel-wheel`、GPL-2.0）は依存としてインストールされるが、importもリンクもしない。
`hadd` は同梱の `obabel` プログラムを別プロセスとして実行する。
