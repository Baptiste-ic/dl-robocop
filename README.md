# dl-robocop

Our task is to detoxify sentences in the same way `paradetox` does.
We are introducing novel methods to augment the dataset and add new loses terms to improve performances

We will use the `paradetox` dataset, augmented with `chatgpt 3.5`, that will try to explain why the imput sentence is toxic or not. This (we hope) will enhance the performances of the model.

We will adopt a teacher-student way of training (TODO: ref paper) to help the teacher model find the complex sementical links faster.

We will also try to use the `Bert` embeding space to keep sementical meaning of the sentence.