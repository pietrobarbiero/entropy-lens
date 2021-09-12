import unittest

import torch
from pytorch_lightning import seed_everything
from torch.nn.functional import one_hot

import entropy_lens as te
from entropy_lens.logic.metrics import test_explanation, complexity
from entropy_lens.logic.nn import entropy


class TestTemplateObject(unittest.TestCase):
    def test_entropy_explain_class_binary(self):
        for i in range(1):
            seed_everything(i)

            # Problem 1
            x0 = torch.zeros((4, 100))
            x = torch.tensor([
                [0, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ], dtype=torch.float)
            x = torch.cat([x, x0], dim=1)
            y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

            layers = [
                te.nn.EntropyLinear(x.shape[1], 10, n_classes=2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(10, 10),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(10, 1),
            ]
            model = torch.nn.Sequential(*layers)

            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            loss_form = torch.nn.CrossEntropyLoss()
            model.train()
            for epoch in range(1001):
                optimizer.zero_grad()
                y_pred = model(x).squeeze(-1)
                loss = loss_form(y_pred, y) + 0.00001 * te.nn.functional.entropy_logic_loss(model)

                loss.backward()
                optimizer.step()

                # compute accuracy
                if epoch % 100 == 0:
                    accuracy = y_pred.argmax(dim=1).eq(y).sum().item() / y.size(0)
                    print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')

            y1h = one_hot(y)
            target_class = 0
            explanation, explanation_raw = entropy.explain_class(model, x, y1h, x, y1h, target_class)
            explanation_complexity = complexity(explanation)
            print(explanation)
            print(explanation_complexity)
            assert explanation == '(feature0000000000 & feature0000000001) | (~feature0000000000 & ~feature0000000001)'
            accuracy, preds = test_explanation(explanation_raw, x, y1h, target_class)
            print(f'Accuracy: {100*accuracy:.2f}%')
            assert accuracy == 1

            target_class = 1
            explanation, explanation_raw = entropy.explain_class(model, x, y1h, x, y1h, target_class)
            explanation_complexity = complexity(explanation)
            print(explanation)
            print(explanation_complexity)
            assert explanation == '(feature0000000000 & ~feature0000000001) | (feature0000000001 & ~feature0000000000)'
            accuracy, preds = test_explanation(explanation_raw, x, y1h, target_class)
            print(f'Accuracy: {100*accuracy:.2f}%')
            assert accuracy == 1

        return

    def test_entropy_multi_target(self):

        # eye, nose, window, wheel, hand, radio
        x = torch.tensor([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=torch.float)
        # human, car
        y = torch.tensor([
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ], dtype=torch.float)
        y1h = y

        layers = [
            te.nn.EntropyLinear(x.shape[1], 10, n_classes=y1h.shape[1], temperature=0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        ]
        model = torch.nn.Sequential(*layers)

        concept_names = ['zero', 'one', 'two', 'three', 'four', 'five']
        target_class_names = ['even', 'odd']

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        loss_form = torch.nn.BCEWithLogitsLoss()
        model.train()

        for epoch in range(6001):
            # train step
            optimizer.zero_grad()
            y_pred = model(x).squeeze(-1)
            loss = loss_form(y_pred, y) #+ 0.001 * te.nn.functional.entropy_logic_loss(model)# + 0.001 * te.nn.functional.l1_loss(model)
            loss.backward()
            optimizer.step()

            # print()
            # print(layers[0].weight.grad[0].norm(dim=1))
            # print(layers[0].weight.grad[1].norm(dim=1))
            # print()

            # compute accuracy
            if epoch % 100 == 0:
                accuracy = (y_pred>0.5).eq(y).sum().item() / (y.size(0)*y.size(1))
                # extract logic formulas
                target_class = 0
                explanation_class_1, exp_raw = entropy.explain_class(model, x, y1h, x, y1h, target_class,
                                                               concept_names=concept_names)
                accuracy1, preds = test_explanation(exp_raw, x, y1h, target_class)
                explanation_class_1 = f'\\forall x: {explanation_class_1} <-> {target_class_names[target_class]}'
                target_class = 1
                explanation_class_2, exp_raw = entropy.explain_class(model, x, y1h, x, y1h, target_class,
                                                               concept_names=concept_names)
                accuracy2, preds = test_explanation(exp_raw, x, y1h, target_class)
                explanation_class_2 = f'\\forall x: {explanation_class_2} <-> {target_class_names[target_class]}'
                print(f'Epoch {epoch}: loss {loss:.4f} train accuracy: {accuracy:.4f}')
                print(f'\tExplanation class 1 ({accuracy1*100:.2f}): {explanation_class_1}')
                print(f'\tExplanation class 2 ({accuracy2*100:.2f}): {explanation_class_2}')
                alpha_norm = layers[0].alpha / layers[0].alpha.max(dim=1)[0].unsqueeze(1)
                print(f'\tAlphas class 1: {alpha_norm[0]}')
                print(f'\tAlphas class 2: {alpha_norm[1]}')
                print()

        return


if __name__ == '__main__':
    unittest.main()
