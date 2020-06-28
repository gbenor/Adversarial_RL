import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

result_dir = Path("results")

def find_minpert_index(history, target):
    images_labeled_as_target = np.array(history["labels"]) == target
    print(f"num of images labeled as target {sum(images_labeled_as_target)}")
    min_pert = (np.array(history["perturbations"])[images_labeled_as_target]).min()
    index = np.where(np.array(history["perturbations"])==min_pert)[0][0]
    assert history["labels"][index] == target, "maybe two images has the same perturbation value"
    return index


def get_steps_and_min_pert(test_name, target):
    steps = []
    min_pert = []
    for f in result_dir.glob(f"*{test_name}*target={target}*pkl*"):
        print(f"target {target} file {f}")
        with f.open('rb') as handle:
            history = pickle.load(handle)
        i = find_minpert_index(history, target=target)
        steps.append(i)
        min_pert.append(history["perturbations"][i])

    return np.array(steps), np.array(min_pert)


def create_statistics(test_name):
    pert_df = pd.DataFrame()
    steps_df = pd.DataFrame()

    # for target in range(10):
    for target in [1]:
        steps, min_pert = get_steps_and_min_pert(test_name, target)
        a = stats.describe(min_pert)
        pert_df = pert_df.append(pd.DataFrame([a], columns=a._fields))
        a = stats.describe(steps)
        steps_df = steps_df.append(pd.DataFrame([a], columns=a._fields))
    return pert_df, steps_df


def join_pert_steps_df(pert_df, steps_df):
    df_dict = {"pert" : pert_df,
               "steps" : steps_df}

    final_col = ['nobs',  'min', 'max', 'mean', 'variance']

    for key in df_dict.keys():
        df_dict[key].reset_index(inplace=True)
        df_dict[key]["min"] = df_dict[key]["minmax"].apply(lambda x: x[0])
        df_dict[key]["max"] = df_dict[key]["minmax"].apply(lambda x: x[1])
        df_dict[key] = df_dict[key][final_col]
        df_dict[key] = df_dict[key].add_prefix(f"{key} ")

    result = df_dict["pert"].join(df_dict["steps"])
    result.drop(columns="steps nobs", inplace=True)
    result.rename(columns={"pert nobs" : "nobs"}, inplace=True)
    result.round()
    result = result.round(decimals=2)
    result = result.astype({'nobs' : int,  'steps min' : int, 'steps max' : int})
    return result


def main(test_name):
    pert_df, steps_df = create_statistics(test_name)
    results = join_pert_steps_df(pert_df, steps_df)

    results.to_csv(result_dir / f"{test_name}_results.csv")
    print(results)


if __name__ == '__main__':
    # tests = ["simple_centers"]
    # tests = ["iterative_centers"]
    # tests = ["iterative_centers_new_init_step"]
    tests = ["RL_policy_small_action_set"]
    for test_name in tests:
        main(test_name)



#
#
# with clip_manager(Path("/tmp/eta/myclip.mp4")) as cm:
#     print(cm)
#     for i in range(20):
#         print(i)
#         img = env.cur_image.reshape(28,28)
#         cm.add_img(img)
#         # plt.imshow(img, cmap="gray")
#         # plt.title(i)
#         # plt.savefig(next(cm))
#         obs, reward, done, _ = env.step(0)
#         env.render()
#
# exit(6)
#     #     env.render()
#     #     predicted_label = np.argmax(obs.predicted_labels)
#     #     if predicted_label == 6:
#     #         image = obs.image.reshape(28,28)
#     #
#     #         plt.imshow(image, cmap='gray')
#     #         plt.savefig("example.pdf", format="pdf", bbox_inches='tight')
#     #         # plt.show()
#     #
#     #         break
#     #
#     # exit(3)
#     #
#
#
#


#
#
#
# images_labeled_as_target = np.array(history["labels"]) == target
# min_pert = (np.array(history["perturbations"])[images_labeled_as_target]).min()
#
#
# # In[14]:
#
#
# plt.imshow(history["images"][19].reshape(28,28), cmap="gray")
# plt.show
#
