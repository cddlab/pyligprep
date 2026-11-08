# 検証記録

初期実装の検証日：2026-11-06。
以下のコマンド表記は現行の引数名に統一している。

## SlurmのCPU割り当てに応じたスレッド数（2026-11-08追記）

- `--threads` 省略時、バッチ直接起動も `srun` 内も `SLURM_CPUS_PER_TASK` を使い、
  CPU affinityのCPU数を上限とする。再開時は前のジョブの保存値より現在の割り当てを優先する。
  明示した値の割り当て超過は計算開始前に拒否する。Slurm外の既定値8は維持。
- 環境変数・affinityを模擬し、1タスク×8 CPU、複数タスクの拒否、srunステップ、
  非連続CPU番号、再開時の割り当て変更、不正値を検証。PySCFへのスレッド数受け渡しも確認。
- 通常テスト **219件成功、15件スキップ**（GPU・AmberTools・reference）。Ruff、変更ファイルのty、
  Slurmスクリプトの構文検査は成功。全体のtyは既存の `dispersion.py` の3件が残る。
- Slurm設定の読み取りで `select/cons_tres`、`CR_CORE`、`task/cgroup,task/affinity`、
  `ConstrainCores=yes` を確認。Slurm計算ジョブの投入・GPU実走は行っていない。

## 最適化・SCFの引数名統一（2026-11-08追記）

- 引数を `--opt-maxcycles`、`--opt-maxstep`、`--scf-maxcycle`、Pythonのフィールドと保存キーを
  `opt_max_cycles`、`opt_max_step`、`scf_max_cycle` に統一。計算条件の値・単位・バックエンドへの渡し方は維持。
- 新名称のCLIから計算処理への受け渡し、再開時の保存設定継承と上書き、旧引数の拒否を確認。
  旧キーを含む保存状態は説明付きエラーで停止し、状態ファイルを書き換えないことも確認。
- 通常テスト **179件成功、15件スキップ**（GPU・AmberTools・reference）。GPUジョブの投入は行っていない。
  Ruffと変更ファイルのtyは成功。全体のtyでは、変更前から存在する `dispersion.py` の3件
  （`dftd3` / `dftd3.pyscf` のimport解決とモジュール代入の型）が残ることを確認。

## 出力構成の簡素化（成果物＋versions.json＋tmp/）と `run --keep-tmp`（2026-11-08追記）

- `manifest.json`・`attempts/`・`result.json`・段階マーカーJSONを廃止し、作業ディレクトリの `state.json`（schema 2）
  1ファイルに統合。各コマンドの `-o` 直下は成果物と `versions.json` のみで、それ以外は `tmp/` に置く。
  再開は試行を増やさず同じ `tmp/` で続きから行う。工程の完了時に `*.chk` を削除する。
- `run` は成功後に `tmp/` を削除し、`--keep-tmp` で保持する。失敗時は常に保持して `run --resume` で再開できる。
- macOS（CPU、AmberTools 26 Homebrew版）で `examples/sin_h.pdb`（charge −2）を実走。記録は `validation/tmp-layout/`。
  - 既定の `run --resname SIN`: 直下は `SIN.{mol2,prep,frcmod}` と `versions.json` の4ファイル、**16 KB**（44秒）。
    `versions.json` に AmberTools26 と `antechamber` / `parmchk2` / `tleap` のパス・SHA-256 を記録。
  - `--keep-tmp`: `tmp/` を含め64ファイル、**496 KB**。`*.chk` は残っていない。
    同じ入力を変更前のコード（HEAD）で実行すると72ファイル、**1.3 MB**。
  - `--keep-tmp` で残した出力への `run --resume` はキャッシュを再利用して0秒で完了。`-r SIM` での再実行は出力工程だけを更新し（1秒）、
    `SIN.*` も残る。
  - `tmp/` を削除した完了済みディレクトリへの `run --resume` は、案内つきのエラー（終了コード2）で停止。
  - `--opt-maxcycles 1` で最適化を未収束のまま停止させると、直下には `tmp/` だけが残り（`last.json`・`restart.chk` を含む）、
    `run --resume --opt-maxcycles 200` で続きから完了して `tmp/` が削除された。
- コードレビューの指摘を受け、`tmp/` 削除失敗を警告扱い、`export --resume` での `resp-source.json` 欠落を案内つきエラー、
  `versions.json` に現在の成果物名を記録、公開の記録をファイルごとに保存するよう修正。
- AmberTools込みの非GPUテスト **137件成功**（GPU・referenceの5件は対象外）。Ruff・tyの指摘ゼロ。

## `run` の成果物を出力ディレクトリ直下へ配置（2026-11-08追記）

- `run` は export ステージの `.mol2` / `.prep` / `.frcmod` を自身の試行ディレクトリへコピーし、
  `result.json` の `files` にハッシュを記録したうえで出力ディレクトリ直下に公開する。`export/` 内の同名ファイルも残す。
- macOS（CPU、AmberTools 26 Homebrew版）で `examples/sin_h.pdb` を `run --resname SIN` で実走。
  `validation/sin-root-output/SIN.{mol2,prep,frcmod}` が `export/` 内のファイルと同一であることを確認。
  `run --resume` でも成功し、試行 `0002` が追加された。ログは `validation/sin-root-output.log`。
- 出力直下を手編集した場合に `run --resume` が上書きせず停止することをテストで確認。
- AmberTools込みの非GPUテスト **131件成功**（GPU・referenceの5件は対象外）。Ruff・tyの指摘ゼロ。

## macOS（Apple Silicon、CPUのみ）対応（2026-11-07追記）

- 環境：macOS 26.7.2（arm64、メモリ18 GB）、Python 3.13.12、PySCF 2.15.0、dftd3 1.7.0、
  RDKit 2026.9.1、Open Babel wheel 3.1.1.23、Homebrew `brewsci/bio` のAmberTools 26.0。
- `pyscf-dispersion` 1.7.0にはmacOS用wheel・sdistがなく、`pyscf[dispersion]` のままでは
  `uv sync` が失敗した。Linuxのみ `pyscf-dispersion`、macOSは `dftd3`（simple-dftd3）を指定し、
  PySCFが呼ぶ `pyscf.dispersion.dftd3` 互換の層（`dispersion.py`）を追加した。
  水分子のB3LYP-D3BJで、分散エネルギーがSCF全エネルギーに含まれること、解析勾配と数値微分の一致を
  テストで確認。
- PyPIのmacOS用PySCF wheelはOpenMPなし（`--threads` が無効）。`pyproject.toml` の `[tool.uv.sources]` と
  `[tool.uv.extra-build-variables]` で、macOSのみPySCFをHomebrew LLVM 23.1.5のclangでソースからビルドするようにした。
  - PyPIのPySCF 2.15.0 sdistは `libxcfun.patch` を含まずビルドに失敗したため、GitHubのv2.15.0リリースアーカイブを使用。
  - Apple clangではCMakeの `find_package(OpenMP)` が警告なしに失敗し、OpenMPなしでビルドされる。
  - CMakeで `Found OpenMP: TRUE (5.1)`。ライブラリはHomebrew LLVMの `libomp.dylib` とAccelerateにリンク。
    `lib.num_threads(8)` が8を返し、警告なし。`dftd3`・Open Babel・RDKitはOpenMPランタイムをリンクせず、混在なし。
  - `uv.lock` ではLinux向けはPyPIのwheel、macOS向けのみURLソース。
  - SINの構造最適化：wheel版98秒、ソース版 `--threads 8` で44秒、`--threads 1` で94秒。
    全体はwheel版98秒、ソース版 `--threads 8` で79秒。`SIN.mol2`・`SIN.prep`・`SIN.frcmod` はwheel版とバイト単位で同一。
  - 設定をプロジェクトに反映後、`.venv` で `run --device cpu --threads 8 --memory-mb 8000 --resname SIN` を再実行し、
    終了コード0、約45秒、OpenMP警告なし。出力3ファイルは `validation/macos-sin/` とバイト単位で同一
    （生成物は `validation/macos-sin-omp/`）。全テスト101件成功、GPUテスト3件はスキップ。
- `examples/sin_h.pdb`（コハク酸ジアニオン、12原子、電荷−2）で
  `run --device cpu --memory-mb 8000 --resname SIN` を実行し、終了コード0、約1分40秒。
  - 最適化：Berny、B3LYP-D3BJ/6-31+G(d,p)、12ステップで収束、E = −455.797550 Eh。
    全ステップでD3(BJ)（`xc=b3lyp, version=d3bj, atm=False`）を適用。
    入力はほぼ最適化済みで、最大変位0.0007 Å。
  - RESP：HF/6-31G*（CPU）、第1・第2段階とも3反復で収束。カルボキシル酸素4個、カルボニル炭素2個、
    メチレン炭素2個、水素4個がそれぞれ等価。
  - 出力：`SIN.mol2`、`SIN.prep`、`SIN.frcmod`。原子名・順番を保持、GAFF2タイプ c / o / c3 / hc。
    MOL2・PREPとも電荷総和 **−2.000000**、残基名 `SIN`。
  - パイプライン外で `tleap` に読み込み直し（`leaprc.gaff2` ＋ FRCMOD ＋ PREP、およびMOL2）、
    両形式で Errors = 0。警告は正味電荷が0でない旨のみ。
  - 生成物は `validation/macos-sin/`。
- `PYLIGPREP_TEST_AMBER=1 PYLIGPREP_TEST_REFERENCE=1` で全テスト **101件成功**、GPUテスト3件はスキップ。
  Ruffの指摘は変更前からの16件のみで、新規の指摘なし。

## 出力ファイル名・残基名の指定（2026-11-06追記）

- `run` / `export` の `--resname` / `-r` で、MOL2・PREP・FRCMODのprefixと
  MOL2・PREP内の残基名を指定。`SIN`、`SiN`、`a1` の大小文字保持を実走確認。
- MOL2のMOLECULE名・ATOM行・SUBSTRUCTURE名、PREPのINT行を検証。
  不一致、無効な名前、再開時の指定継承・変更、手編集した出力の保護をテスト。
- 残基名は出力工程に適用し、最適化・RESPへ渡す分子と元のRESP結果を保持。
- AmberTools 26と元のPyRESP参照実装を含む非GPUテスト **97件成功**。
  GPUテスト3件は対象外。Ruff静的チェック・整形確認が成功。
  非GPUテストの全体カバレッジ85%、出力モジュール93%、CLI 98%、一括処理100%。
  記録は `validation/resname-coverage.json`。
- SINの完了済みRESPから `validation/sin-resname/SIN.{mol2,prep,frcmod}` を作成。
  12原子、MOL2の電荷総和 **−2.000000**、MOL2/PREP両形式のLEaP Errors=0。
  手編集済み `sin_run/export/ligand.{mol2,prep,frcmod}` のハッシュ不変を確認。
  検証記録は `validation/resname-sin-verification.json`。
  構造最適化・ESP・RESPの再計算、GPUジョブの投入は行っていない。

## 出力電荷の5桁化とPREP拡張子（2026-11-06追記）

- 電荷を5桁で四捨五入し、6桁目を0として出力。半端値は絶対値の大きい側へ丸める。
- 1e-5 e単位の整数計算により、指定総電荷と実際のファイルの十進数総和を一致させる。
- 第2段階の等価拘束と、固定原子に引き継がれた第1段階の等価拘束を維持。
- 単一候補を飛ばす場合、組を同時に動かす場合、逆方向・複数単位の補正、
  等価性と整数総和が数学的に両立しない場合の明示的失敗をテスト。
- 100組の再現可能なランダム入力でも総和・等価性を検証。
- PREPI形式を `.prep` に出力し、MOL2からの変換・往復変換・LEaPを実走。
- 旧版の完了済みexportを新しい設定で再実行し、過去の試行を保持することを確認。
- GPU以外の全テスト **78件成功**、GPUテスト3件は対象外。量子計算の変更なし。
- 新規電荷丸めモジュールのステートメントカバレッジ **98%**。
  未通過2行は通常の有効入力では到達しない内部整合性チェック。
- Ruff静的チェック・整形確認、wheel/sdistのビルドが成功。
  新ビルドは `dist/charge-export-v1/`、カバレッジ記録は
  `validation/charge-export-coverage.json`。
- 実リガンドligcl3の保存済みRESP結果を再利用して再出力し、78原子の電荷総和が
  MOL2/PREPとも厳密に **−1.000000**、等価組の電荷一致、LEaP Errors=0を確認。
  FRCMODは変更前と同一。HF/ESP/RESPの再計算、GPUジョブの投入は行っていない。

以下は初期実装時の記録。

## 環境

uvで作成した独立環境を使用。手動ビルド済みGPU4PySCF環境は変更していない。

- Python 3.13.11、PySCF 2.15.0、GPU4PySCF CUDA12 wheel 1.9.1
- CuPy CUDA12 13.4.1、cuTENSOR 2.2.0、GPU libxc 0.9.0
- PySCF dispersion 1.7.0、PyBerny 0.7.0、geomeTRIC 1.1.1
- RDKit 2026.9.1、Open Babel wheel 3.1.1.23
- AmberTools 26、Slurm `q3`、RTX 4090

## 実施内容

- 中性・陰イオン・陽イオン・塩素を含む分子の水素付加と原子名保持。
- 不正PDB、電荷／多重度不一致、古いサイドカー、改変された出力の拒否。
- CPUでのHF単点計算とESP積分の独立比較。CPU構造最適化は実走していない。
- 元のPyRESP_GENとの制約生成比較、元のPyRESPとの二段階電荷比較。
- RKS／UKSの全4方向のチェックポイント読み込みと不正データ拒否。
- フィット失敗後のESP再利用と、各工程のキャッシュ・再開・CLI。
- AmberToolsでのGAFF2割り当て、PREPI往復変換、parmchk2、両形式のLEaP検証。
- GPUでのHF/ESPとCPU結果の一致、B3LYP-D3BJ/6-31+G(d,p)最適化、
  1ステップで未収束にした後の再開、geomeTRIC、RESP、最終ファイル出力。

## Slurm

最初のジョブ `257077` はCUDA 12.9とAmberTools要求の12.8のモジュール競合で、
計算前に停止。スクリプトをAmberTools 26によるCUDAロードに統一した。
ジョブ `257078` は `q3` で正常終了し、GPUテスト2件が成功（約43秒）。
CLI全工程とカバレッジを含む最終ジョブ `257085` も `q3` で正常終了した
（exit code 0、約54秒、GPUテスト3件成功）。`run` の一括処理と、完了済み
`run --resume` が最適化を再実行しないことも確認した。

## 最終結果

- GPU以外：50件成功。AmberToolsと元のPyRESPを使う参照テストを含む。
- GPU：3件成功。上記ジョブで実施。
- 合算ステートメントカバレッジ：1160 / 1282、90.48%。
- Ruffの静的チェック・整形確認、Slurmスクリプトのbash構文確認が成功。
- 新規ファイルの空白差分チェックが成功。既存 `ligcl3_opt.py` のハッシュ不変を確認。
- uvによるwheelとsdistのビルドが成功。ライセンス・ソース・テスト等の同梱と、
  仮想環境・キャッシュ・計算結果が配布物に含まれないことを確認。
- wheelを独立環境へインストールし、ソースディレクトリを参照せずに
  `hadd` と `check` を実行できることを確認。

GPU4PySCF 1.9.1由来のSciPy非推奨API警告が発生するが、検証結果は成功。

## 実際のligcl3入力に関する注意

既存 `ligcl3.pdb` をテンプレートなしで処理した結果は C30H48O3、形式電荷0となった。
既存の最適化計算の C30H45O3、電荷−1とは異なる。
これはPDB座標だけから結合次数やプロトン化状態を一意に復元できない実例であり、
この自動推定構造を既存計算と同じリガンド状態として使用してはいけない。
目的の状態を指定するSMILESまたはSDF/MOL/MOL2テンプレートを使用し、
水素付加後の構造を確認する。

既存 `ligcl3_opt.py` の水素を含むATOM座標からRDKitで結合を割り当て、
`validation/ligcl3_reference.sdf` を作成した。このテンプレートを指定した
`hadd` の結果 `validation/ligcl3_h_templated.pdb` は78原子、C30H45O3、電荷−1だった。
元のPDB原子名と重原子座標を保持する。新規水素の座標は再生成されるため、
既存最適化結果と同一座標ではない。等価な原子のテンプレート対応についてRDKitの
複数候補警告も出るため、これらはユーザー確認用の参考ファイルとして扱う。
実リガンドの最適化やRESP再計算は投入していない。

## 検証の範囲

初版は単一分子・単一残基・単一配座の H/C/N/O/F/P/S/Cl に限定する。
全有機分子のプロトン化・等価原子判定・GAFF2パラメータの妥当性を保証するものではない。
テストの生ログと生成物は開発環境の `validation/` に保存し、配布物には含めない。

## 2026-11-08：最適化専用の重なり行列閾値

基準コミット `2b2d0e3` に `--opt-overlap-threshold` を追加し、`quantum.py` のSCF回数名を
`scf_max_cycle` / `--scf-maxcycle` に統一した。閾値未指定時の既定動作とキャッシュ互換性を維持する。

- 追加テストのREDを確認後、通常の全体テストは251件成功、17件スキップ。
- Slurmジョブ `258442`（q4、fdn15、GPU 1基）は3件成功、終了コード0。
  RKS/UKSのNewton SCFから勾配scannerを2構造で実行し、GPU上の閾値保持と直交化を確認。
  既存のCPU/GPU HF-ESP一致テストも成功。
- CPUでもHF/DFT、制限／非制限、scanner更新、閾値の境界、異常入力、CLIの受渡し、
  再開時の継承／上書き、キャッシュ無効化、RESPへの非伝播を検証した。
- Ruffチェックと整形確認、変更したPythonファイルのtyチェックは成功。
  全体のtyチェックには既存の `dispersion.py` の3件（Linuxでの `dftd3` 解決2件と代入型1件）が残る。
- 検証は一時コピーで実施。GPUログは
  `/home/moriwaki/tomato/scf/validation/pyligprep-overlap-1ocziobi/gpu-258442.out` に保存。
  CL4の構造最適化そのものは再投入しておらず、CL4の収束改善は未検証。
