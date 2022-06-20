import pandas as pd
import numpy as np
from sklearn import linear_model


class Race:
    def __init__(self):
        self.race_numbers = list(range(1, 5))
        self.results_link = "data/processed/Manual times.xlsx"
        self.drivers = pd.read_excel(
            self.results_link, sheet_name="Race 1", index_col=0,
        ).columns

    def get_race_results(self):
        race_results = []
        for race_number in self.race_numbers:
            race_results.append(
                pd.read_excel(
                    self.results_link,
                    sheet_name="Race " + str(race_number),
                    index_col=0,
                )
            )
        return pd.concat(
            race_results, keys=[1, 2, 3, 4], names=["Race Number"]
        ).reset_index()

    def get_race_finish_times(self, race_results=None):
        if race_results is None:
            race_results = self.get_race_results()
        finish_times = pd.DataFrame(index=self.race_numbers, columns=self.drivers)
        for race_number in self.race_numbers:
            finish_times.loc[race_number] = race_results[
                race_results["Race Number"] == race_number
            ][self.drivers].sum()

        return finish_times

    def get_lap_count(self, for_drivers=True, race_results=None):
        if race_results is None:
            race_results = self.get_race_results()
        if for_drivers:
            lap_count = pd.DataFrame(index=self.race_numbers, columns=self.drivers)
            for race_number in self.race_numbers:
                lap_count.loc[race_number] = race_results[
                    race_results["Race Number"] == race_number
                ][self.drivers].count()

        return lap_count

    def get_mean_lap(self, for_drivers=True):
        return self.get_race_finish_times() / self.get_lap_count(for_drivers=True)

    def get_race_ranking(self):
        mean_lap = self.get_mean_lap()
        return mean_lap.rank(axis=1)

    def get_race_ranking_summary_table(self):
        rankings = self.get_race_ranking()
        points = Points()
        df = pd.DataFrame()
        df["Mario Kart Points"] = rankings.replace(points.mario_kart()).sum()
        df["F1 Race Points"] = rankings.replace(points.f1_position()).sum()
        df["F1 Sprint Points"] = rankings.replace(points.f1_sprint()).sum()
        df.sort_values(by="Mario Kart Points", ascending=False, inplace=True)
        return df

    def get_fastest_lap_ranking_summary_table(self):
        fastest_laps = self.get_fastest_lap_ranking()
        points = Points()
        df = pd.DataFrame()
        df["Mario Kart Points"] = fastest_laps.replace(points.mario_kart()).sum()
        df["F1 Race Points"] = fastest_laps.replace(points.f1_position()).sum()
        df["F1 Sprint Points"] = fastest_laps.replace(points.f1_sprint()).sum()
        df.sort_values(by="Mario Kart Points", ascending=False, inplace=True)
        return df

    def get_fastest_lap(self):
        race_results = self.get_race_results()
        fastest_laps = pd.DataFrame(index=self.race_numbers, columns=self.drivers)
        for race_number in self.race_numbers:
            fastest_laps.loc[race_number] = race_results[
                race_results["Race Number"] == race_number
            ][self.drivers].min()
        return fastest_laps

    def get_fastest_lap_ranking(self):
        fastest_laps = self.get_fastest_lap()
        return fastest_laps.rank(axis=1)

    def get_filtered_lap_mean(self):
        race_results = self.get_race_results()
        race_results[race_results > 35] = np.nan
        return self.get_race_finish_times(race_results) / self.get_lap_count(
            True, race_results
        )

    def get_filtered_lap_ranking(self):
        filtered_lap_mean = self.get_filtered_lap_mean()
        return filtered_lap_mean.rank(axis=1)

    def get_kart_numbers(self):
        return pd.read_excel(self.results_link, sheet_name="Karts", index_col=0)

    def flatten(self, df):
        return df.to_numpy().flatten()

    def get_kart_fastest_lap(self):
        kart_numbers_list = self.flatten(self.get_kart_numbers())
        fastest_laps_list = self.flatten(self.get_fastest_lap())

        kart_all_fastest_laps = pd.DataFrame(
            index=kart_numbers_list, data=fastest_laps_list, columns=["Fastest Lap"]
        )
        kart_all_fastest_laps["Fastest Lap"] = pd.to_numeric(
            kart_all_fastest_laps["Fastest Lap"]
        )

        kart_fastest_lap = pd.DataFrame(
            index=np.unique(kart_numbers_list), columns=["Fastest Lap"]
        )
        for kart_number in kart_fastest_lap.index:
            kart_times = kart_all_fastest_laps.loc[kart_number]
            if len(kart_times) > 1:
                kart_fastest_lap.loc[kart_number] = kart_times.min()
            else:
                kart_fastest_lap.loc[kart_number] = kart_times.iloc[0]

        return kart_fastest_lap

    def get_start_positions(self):
        race_results = self.get_race_results()
        drivers = ["Pat", "Jeremy", "Eric", "Ryan", "Arjun", "Sean", "Anthony"]
        return (
            race_results[drivers][(race_results["Lap"] == 1)]
            .rank(axis=1)
            .set_index(pd.Index([1, 2, 3, 4]))
        )

    def make_results_table(self):
        df = pd.DataFrame(
            columns=[
                "Race Number",
                "Kart Number",
                "Driver",
                "Mean Lap",
                "Filtered Mean Lap",
                "Fastest Lap",
                "Start Position",
            ]
        )
        df["Race Number"] = [1] * 7 + [2] * 7 + [3] * 7 + [4] * 7
        df["Kart Number"] = self.flatten(self.get_kart_numbers())
        df["Driver"] = self.drivers.to_list() * len(self.race_numbers)
        df["Mean Lap"] = self.flatten(self.get_mean_lap())
        df["Filtered Mean Lap"] = self.flatten(self.get_filtered_lap_mean())
        df["Fastest Lap"] = self.flatten(self.get_fastest_lap())
        df["Start Position"] = self.flatten(self.get_start_positions())
        return df


class Regression:
    def __init__(self, df, y_name):
        self.df = df
        self.y_name = y_name

    def one_hot_encode(self, column):
        return pd.concat(
            [
                self.df.drop(columns=column),
                pd.get_dummies(self.df[column], prefix=column),
            ],
            axis=1,
        )

    def add_one_hot_encode(self):
        for column in ["Driver", "Kart Number"]:
            self.df = self.one_hot_encode(column)
        return self.df

    def ridge_regression(self, alpha=0.1):
        self.df = self.add_one_hot_encode()
        X = self.df.drop(columns=[self.y_name])
        y = self.df[self.y_name]
        model_r = linear_model.Ridge(alpha=alpha)
        model_r.fit(X, y)
        print(model_r.score(X, y))
        return model_r

    def lasso_regression(self, alpha=0.1):
        self.df = self.add_one_hot_encode()
        X = self.df.drop(columns=[self.y_name])
        y = self.df[self.y_name]
        model_l = linear_model.Lasso(alpha=alpha)
        model_l.fit(X, y)
        return model_l

    def elastic_net_regression(self, alpha=0.1, l1_ratio=0.5):
        self.df = self.add_one_hot_encode()
        X = self.df.drop(columns=[self.y_name])
        y = self.df[self.y_name]
        model_e = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model_e.fit(X, y)
        return model_e

    def get_coefficients(self, model):
        return pd.DataFrame(
            model.coef_, index=self.df.drop(columns=[self.y_name]).columns
        )

    def get_intercept(self, model):
        return model.intercept_

    def get_predictions(self, model):
        return model.predict(self.df.drop(columns=[self.y_name]))

    def get_residuals(self, model):
        return self.df[self.y_name] - self.get_predictions(model)

    def get_residual_sum_of_squares(self, model):
        return np.sum(self.get_residuals(model) ** 2)

    def get_mean_squared_error(self, model):
        return np.mean((self.df[self.y_name] - self.get_predictions(model)) ** 2)

    def get_mean_absolute_error(self, model):
        return np.mean(np.abs(self.df[self.y_name] - self.get_predictions(model)))

    def get_r2_score(self, model):
        return model.score(self.df.drop(columns=[self.y_name]), self.df[self.y_name])


class Points:
    # def __init__(self) -> None:

    def mario_kart(self):
        # from https://mariokart.fandom.com/wiki/Driver%27s_Points
        return {1: 15, 2: 12, 3: 10, 4: 9, 5: 8, 6: 7, 7: 6}

    def f1_position(self):
        # from https://f1.fandom.com/wiki/Points
        return {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6}

    def f1_sprint(self):
        # from https://f1.fandom.com/wiki/Points
        return {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2}
