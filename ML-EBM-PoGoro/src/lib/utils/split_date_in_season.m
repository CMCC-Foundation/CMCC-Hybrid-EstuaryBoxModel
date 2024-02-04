function dataset = split_date_in_season(dataset, years)
    % Define the start and end dates for each season
    for i=1:numel(years)
        start_year = datetime(years(i),01,01);
        spring_start = datetime(years(i),03,20);
        summer_start = datetime(years(i),06,21);
        autumn_start = datetime(years(i),09,22);
        winter_start = datetime(years(i),12,21);
        end_year = datetime(years(i),12,31);
    
        dataset.Season(isbetween(dataset.Date, start_year, spring_start, "openright")) = "Winter";
        dataset.Season(isbetween(dataset.Date, spring_start, summer_start, "openright")) = "Spring";
        dataset.Season(isbetween(dataset.Date, summer_start, autumn_start, "openright")) = "Summer";
        dataset.Season(isbetween(dataset.Date, autumn_start, winter_start, "openright")) = "Autumn";
        dataset.Season(isbetween(dataset.Date, winter_start, end_year)) = "Winter";
    end
    dataset.Season = categorical(dataset.Season);
end
