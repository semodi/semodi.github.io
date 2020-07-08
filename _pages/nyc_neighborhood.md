---
layout: splash
title:  "NYC Neighborhoods"
permalink: /nyc-neighborhoods/
date:   2020-07-05
--- 

# Finding (Pareto) Optimal Neighborhoods in NYC

## Introduction

Finding a place to live is never easy, this is especially true in a mega-city like New York. The options seem endless, and different trade-offs need to be considered when hunting for apartments: Do I want to live closer to my workplace but pay a higher rent, or do I possibly want to move to a more quiet, safer, residential neighborhood sacrificing valuable time during my commute?

When it comes to choosing the right place to live, every individual will have different priorities, so there is no "one size fits all" solution. In terms of formal decision-making theory, this problem can be cast as a multi-objective or Pareto optimization.
As the name suggests, rather than optimizing a variable (in this case the neighborhood) to minimize a single cost-function (i.e. rent) we want to simultaneously optimize several cost-functions. While not entirely accurate in the mathematical sense we will refer to these cost-functions loosely as metrics in the remainder of this report and we will specify the metrics used in the following section.

The goal of this report will be to find Pareto efficient (PE) (aka Pareto optimal) NYC Neighborhoods. The easiest way to understand Pareto efficiency is in terms of a negative example: Let us assume we want to find a neighborhood that is both cheap and safe. The metrics are therefore median rent and crime rate. If a neighborhood is **not** Pareto efficient, we can always find a different neighborhood that improves at least one of the metrics while not impairing any other ones. Conversely, if this is not possible, the neighborhood is called Pareto efficient.

![pareto_example_cleaned.png](/assets/img/nyc/pareto_example_cleaned.png)

***Figure 1.*** *Pareto efficiency for synthetic dataset. The point highlighted in green is not PE as both Cost 1 and Cost 2 can be decreased by moving along the path indicated by the arrow.*

The benefits of this approach are clear, as simply finding the "best" neighborhood would require us to specify the relative importance of rent and crime rate. This relative importance, however, is highly subjective and will differ from person to person. 
Pareto efficiency serves as an objective tool to help the apartment-seeker find their best fit. Based on their personal preferences, they can go through the list of PE neighborhoods and choose the one closest to their liking. It should be noted that, from an optimization standpoint, it does not make sense to pick a non-PE neighborhood.


## Data

The metrics and associated data used in this report are the following

1. **Safety**
    
    To calculate the average safety of a given neighborhood I will combine data on the number of arrests made and the number of shootings. Both datasets are availabe at [NYC Open Data](https://opendata.cityofnewyork.us/).
    
1. **Rent**

    To gauge rent prices in every neighborhood, I decided to analyze the median rent for a one bedroom apartment. The data was obtained from [StreetEasy](https://streeteasy.com/blog/data-dashboard/?agg=Total&metric=Inventory&type=Sales&bedrooms=Any%20Bedrooms&property=Any%20Property%20Type&minDate=2010-01-01&maxDate=2020-05-01&area=Flatiron,Brooklyn%20Heights)

1. **Venue Density**

    Using the Foursquare API, in particular the "explore" endpoint, one can estimate the venue density in a given neighborhood. The venue density will be defined as the number of venues returned by Foursquare in a 500m radius around the neighborhood center.
    
1. **Distance from Subway**

    Combining [location data](https://data.cityofnewyork.us/Transportation/Subway-Entrances/drex-xx56) on subway entrances with [NYC geodata](https://data.beta.nyc/dataset/pediacities-nyc-neighborhoods) one can determine the average distance to the closest subway entrance for each neighborhood.

1. **Distance from Midtown**

    Given the shorter commute time, it might be desirable for some people to live as close as possible to their workplace. Both, the Financial District, Midtown Manhattan are the centers of economic activity in New York. I have chosen the latter to calculate this distance metric. 
    
    


    
