# Code for paper "Towards Fair and Robust Classification"

## General notes (**important**):
- All codes are implemented in Python 3.7.9 with PyTorch 1.7.1;
- Due to further change to the paper, for all codes below, the corresponding experiment is referred to "paper.pdf" in current directory as a snapshot;
- All experimental scripts are divide into 3 code blocks:
	- Implementation: The code that implements the experiment with specific algorithm, these script takes arguments of a single setting from command line and only execute and output the result for that setting. For example ``python InProcess.py adult race FGSM 0.1 0.4`` will execute in-processing algorithm with adult as the dataset, race as the senstive attribute, FGSM as the robustness against, lambda_R=0.1 and lambda_F=0.4.
	- Batching script: These scripts has prefix ``exec_``, e.g., ``exec_InProcess.py``, which are used to enmuerate all settings for the implementation to execute. The scripts can resume the progress, i.e., it can automatically detect which settings have been exeucted, skip them and start from the next uncalulated setting.
	- Result script: These scripts convert the raw output of the implementation into parsable results, e.g., tabular data, pdf figures, etc.
	- ** In a word, the implementation part can be ignored unless there are some erros, and the execution sequence is: (1) Run batching scripts; and (2) Run result scripts and get the result. **


## Existing Approaches (Figure 2)

- Implementation: ``ExistingCombosFnR.py``
- Batching script: ``exec_ExistingCombosFnR.py``
- Result script: ``./result/existings/parse_FnR.py``, the output contains three lines and need to be filled in the pgfplot script in corrseponding ``.tex`` file in the paper source.

## Heatmap of in-processing (Figure 3)

- Implementation: ``InProcess.py``
- Batching script: ``exec_InProcess.py``
- Result script: ``./result/inproc/parse_Heatmap.py``

## Angle of in-processing gradient (Figure 4)

- Implementation: ``InProcess.py``
- Batching script: ``exec_InProcess.py``
- Result script: ``./result/inproc/parse_Angle.py``, the output contains three lines and need to be filled in the pgfplot script in corrseponding ``.tex`` file in the paper source.
