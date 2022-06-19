import pandas as pd


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

    def get_kart_numbers(self):
        return pd.read_excel(self.results_link, sheet_name="Karts", index_col=0)

    def get_race_finish_times(self):
        race_results = self.get_race_results()
        finish_times = pd.DataFrame(index=self.race_numbers, columns=self.drivers)
        for race_number in self.race_numbers:
            finish_times.loc[race_number] = race_results[
                race_results["Race Number"] == race_number
            ][self.drivers].sum()

        return finish_times

    def get_lap_count(self, for_drivers=True):
        race_results = self.get_race_results()
        if for_drivers:
            lap_count = pd.DataFrame(index=self.race_numbers, columns=self.drivers)
            for race_number in self.race_numbers:
                lap_count.loc[race_number] = race_results[
                    race_results["Race Number"] == race_number
                ][self.drivers].count()

        return lap_count

    def get_average_lap(self, for_drivers=True):
        return self.get_race_finish_times() / self.get_lap_count(for_drivers=True)

    def get_race_ranking(self):
        average_lap = self.get_average_lap()
        return average_lap.rank(axis=1)


class Points:
    # def __init__(self) -> None:

    def mario_kart_points(self):
        # from https://mariokart.fandom.com/wiki/Driver%27s_Points
        return {1: 15, 2: 12, 3: 10, 4: 9, 5: 8, 6: 7, 7: 6}

    def f1_position_points(self):
        # from https://f1.fandom.com/wiki/Points
        return {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6}

    def f1_sprint_points(self):
        # from https://f1.fandom.com/wiki/Points
        return {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2}
