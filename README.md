# Project 2 - Ames Housing Data and Kaggle Challenge

## Problem Statement

In real estate industry, housing prices are of great interest for both buyers and sellers. As a leader in real estate consulting, RF Real Estate Consulting Firm offer services related to housing market research and valuation for houseowners and buyers. The valuation of house price is based on house features; however, so far in the property market in Iowa, there is a lack of trustworthy and professional price prediction model out there. 

Hence, the goal of this project is to create a regression model that is able to accurately predict the house sales price given the house features. 

To develop the most accurate model, I will be evaluating 3 models: 1. `LinearRegression`, 2. `LassoCV` and 3. `RidgeCV`. Performance metrics will be used to compare which model is the most appropriate in this case and the two performance metrics are: 

1. R-squared (R2) : R2 score measures the proportion of variation in the dependent variable (house prices) that can be attributed to the independent variable (house features). A higher R2 score indicates a better fit for the model. In this project, we will evaluate which model has the highest R2 score, out of 1. 

2. Root-mean-square error (RMSE) : RMSE measures the difference between predicted values and the actual values. The RMSE of the train and test datasets should be very similar if a good model is built. Hence, here we are looking for the model that can minimize RMSE.

With the above information, our company will be able to generate more accurate house price prediction given those highly correlated housefeatures. Eventually, it benefits all the stakeholders, including our direct users - real estate consultants, and indirect users - sellers and buyers. Our real estate consultants can make use of the predicted house prices to better our advise clients accurately with credibility and enhance our company reputation. 

## Executive Summary

RF Real Estate Consulting company has been offering real estate consulting services in purchasing adn selling houses in Iowa. The most asked question among all stakeholders, include consulting professionals, buyers and sellers is ' how much?'. The house price is the most important indicator that affects the decision making. 

We have worked closely with developers, investors and institutions such as banks and governments to be the bedrock for all their real estate needs. Developing a model to predict housing sales price will allow us to have a more in-depth understanding of local housing market in Ames, so that our clients are able to reduce costs, create value and improve the investment performance.

Due to the pandemic situation since the beginning of the year, there are more uncertainties in the property market in Iowa. Our company is looking for a reliable model that can help predict the house prices more accurately, so that we are able to better advise our clients if they are selling or buying at the ideal price. Only if we can maintain our consulting services professional and trustworthy, we will be able to build the trust with our clients and generate more revenue for our company. 

Our company will count on this house prediction model to achieve three main objectives:

1. Enhance company reputation: 

    The house price prediction model could be a huge boost to our clients's confidence. There is always a fierce competition among the real estate consulting services in Iowa. In order to stand out, it is essential to provide more professional and more credible service that our competitors can't. Hence, we will also make use of this model as one of our selling point in the marketing campaigns to strenthen our branding, so as to attract more new clients and retain the existing clients. 
    
    
2. Build trust with clients and develop long term customer relationship:

    Apart from marketing strategy, we will also enhance the customer relationship management making use of this model. Since this model is exclusively created and used by our company, we believe it will enable us to build trust with clients and develop long term customer relationship. In long term, our company will be able to build a client base that stablizes company's source of business and revenue. 


3. Generate more revenue: 

    The ultimate goal of developing the model and adjusting marketing and operation strategies is to generate more revenue for the company.  


Our company looks forward to making use of the model to align our strategy with business goals and help our business to be more successful. 


### Contents:
1. [Data Import](#Data-Import)
2. [Data Dictionary](#Data-Dictionary)
3. [Data Cleaning](#Data-Cleaning)
4. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
5. [Preprocessing and Modeling](#Preprocessing-and-Modeling)
6. [Descriptive and Inferential Statistics](#Descriptive-and-Inferential-Statistics)
7. [Predict Test Dataset](#Predict-Test-Dataset)
7. [Conclusions and Recommendations](#Conclusions-and-Recommendations)



## Data Dictionaries

|**Feature**|Type|Description|Values|
| :---|:---|:---|:---|
|**id**|*int*|ID|Integer|
|**ms_subclass**|*int*|The building class|20 1-STORY 1946 & NEWER ALL STYLES <br> 30 1-STORY 1945 & OLDER <br> 40 1-STORY W/FINISHED ATTIC ALL AGES <br> 45 1-1/2 STORY - UNFINISHED ALL AGES <br> 50 1-1/2 STORY FINISHED ALL AGES <br> 60 2-STORY 1946 & NEWER <br> 70 2-STORY 1945 & OLDER <br> 75 2-1/2 STORY ALL AGES <br> 80 SPLIT OR MULTI-LEVEL <br> 85 SPLIT FOYER <br> 90 DUPLEX - ALL STYLES AND AGES <br> 120 1-STORY PUD (Planned Unit Development) - 1946 & NEWER <br> 150 1-1/2 STORY PUD - ALL AGES <br>  160 2-STORY PUD - 1946 & NEWER <br> 180 PUD - MULTILEVEL - INCL SPLIT LEV/FOYER <br> 190 2 FAMILY CONVERSION - ALL STYLES AND AGES <br>|
|**ms_zoning**|*object*|Identifies the general zoning classification of the sale|A Agriculture <br> C Commercial <br> FV Floating Village Residential <br> I Industrial <br> RH Residential High Density <br> RL Residential Low Density <br> RP Residential Low Density Park <br> RM Residential Medium Density|
|**lot_frontage**|*float*|Linear feet of street connected to property|feet|
|**lot_area**|*int*|Lot size in square feet|square feet|
|**street**|*object*|Type of road access to property|Grvl Gravel <br> Pave Paved|
|**alley**|*object*|Type of alley access to property|Grvl Gravel <br> Pave Paved <br> NA No alley access|
|**lot_shape**|*object*|General shape of property|Reg Regular <br> IR1 Slightly irregular <br> IR2 Moderately Irregular <br> IR3 Irregular|
|**land_contour**|*object*|Flatness of the property|Lvl Near Flat/Level <br> Bnk Banked - Quick and significant rise from street grade to building <br> HLS Hillside - Significant slope from side to side <br> Low Depression|
|**utilities**|*object*|Type of utilities available|AllPub All public Utilities (E,G,W,& S) <br> NoSewr Electricity, Gas, and Water (Septic Tank) <br> NoSeWa Electricity and Gas Only <br> ELO Electricity only|
|**lot_config**|*object*|Lot configuration|Inside Inside lot <br> Corner Corner lot <br> CulDSac Cul-de-sac <br>  FR2 Frontage on 2 sides of property <br>  FR3 Frontage on 3 sides of property|
|**land_slope**|*object*|Slope of property|Gtl Gentle slope <br> Mod Moderate Slope <br> Sev Severe Slope|
|**neighborhood**|*object*|Physical locations within Ames city limits|Blmngtn Bloomington Heights <br> Blueste Bluestem <br> BrDale Briardale <br> BrkSide Brookside <br> ClearCr Clear Creek <br> CollgCr College Creek <br> Crawfor Crawford <br> Edwards Edwards <br> Gilbert Gilbert <br> IDOTRR Iowa DOT and Rail Road <br> MeadowV Meadow Village <br> Mitchel Mitchell <br> Names North Ames <br> NoRidge Northridge <br> NPkVill Northpark Villa <br> NridgHt Northridge Heights <br> NWAmes Northwest Ames <br> OldTown Old Town <br> SWISU South & West of <br> Iowa State University <br> Sawyer Sawyer <br> SawyerW Sawyer West <br> Somerst Somerset <br> StoneBr Stone Brook <br> Timber Timberland <br> Veenker Veenker|
|**condition_1**|*object*|Proximity to main road or railroad|Artery Adjacent to arterial street <br> Feedr Adjacent to feeder street <br> Norm Normal <br> RRNn Within 200' of North-South Railroad <br> RRAn Adjacent to North-South Railroad <br> PosN Near positive off-site feature--park, greenbelt, etc. <br> PosA Adjacent to postive off-site feature <br> RRNe Within 200' of East-West Railroad <br> RRAe Adjacent to East-West Railroad|
|**condition_2**|*object*|Proximity to main road or railroad (if a second is present|Artery Adjacent to arterial street <br> Feedr Adjacent to feeder street <br> Norm Normal <br>  RRNn Within 200' of North-South Railroad <br>  RRAn Adjacent to North-South Railroad <br>  PosN Near positive off-site feature--park, greenbelt, etc. <br>  PosA Adjacent to postive off-site feature <br>  RRNe Within 200' of East-West Railroad <br>  RRAe Adjacent to East-West Railroad|
|**bldg_type**|*object*|Type of dwelling|1Fam Single-family Detached <br> 2FmCon Two-family Conversion; originally built as one-family dwelling <br> Duplx Duplex <br> TwnhsE Townhouse End Unit <br> TwnhsI Townhouse Inside Unit|
|**house_style**|*object*|Style of dwelling|1Story One story <br> 1.5Fin One and one-half story: 2nd level finished <br> 1.5Unf One and one-half story: 2nd level unfinished <br> 2Story Two story <br> 2.5Fin Two and one-half story: 2nd level finished <br> 2.5Unf Two and one-half story: 2nd level unfinished <br> SFoyer Split Foyer <br> SLvl Split Level|
|**overall_qual**|*int*|Overall material and finish quality|10 Very Excellent <br> 9 Excellent <br> 8 Very Good <br> 7 Good <br> 6 Above Average <br> 5 Average <br> 4 Below Average <br> 3 Fair <br> 2 Poor <br> 1 Very Poor|
|**overall_cond**|*int*|Overall condition rating|10 Very Excellent <br> 9 Excellent <br> 8 Very Good <br> 7 Good <br> 6 Above Average <br> 5 Average <br> 4 Below Average <br> 3 Fair <br> 2 Poor <br> 1 Very Poor|
|**year_built**|*int*|Original construction date|Year|
|**year_remod/add**|*int*|Remodel date (same as construction date if no remodeling or additions)|Year|
|**roof_style**|*object*|Type of roof|Flat Flat <br> Gable Gable <br> Gambrel Gabrel (Barn) <br> Hip Hip <br> Mansard Mansard <br> Shed Shed|
|**roof_matl**|*object*|Roof material|ClyTile Clay or Tile <br> CompShg Standard (Composite) Shingle <br> Membran Membrane <br> Metal Metal <br> Roll Roll <br> Tar&Grv Gravel & Tar <br> WdShake Wood Shakes <br> WdShngl Wood Shingles|
|**exterior_1st**|*object*|Exterior covering on house|AsbShng Asbestos Shingles <br> AsphShn Asphalt Shingles <br> BrkComm Brick Common <br> BrkFace Brick Face <br> CBlock Cinder Block <br> CemntBd Cement Board <br> HdBoard Hard Board <br> ImStucc Imitation Stucco <br> MetalSd Metal Siding <br> Other Other <br> Plywood Plywood <br> PreCast PreCast <br> Stone Stone <br> Stucco Stucco <br> VinylSd Vinyl Siding <br> Wd Sdng Wood Siding <br> WdShing Wood Shingles|
|**exterior_2nd**|*object*|Exterior covering on house (if more than one material)|AsbShng Asbestos Shingles <br> AsphShn Asphalt Shingles <br> BrkComm Brick Common <br> BrkFace Brick Face <br> CBlock Cinder Block <br> CemntBd Cement Board <br> HdBoard Hard Board <br> ImStucc Imitation Stucco <br> MetalSd Metal Siding <br> Other Other <br> Plywood Plywood <br> PreCast PreCast <br> Stone Stone <br> Stucco Stucco <br> VinylSd Vinyl Siding <br> Wd Sdng Wood Siding <br> WdShing Wood Shingles|
|**mas_vnr_type**|*object*|Masonry veneer type|BrkCmn Brick Common <br> BrkFace Brick Face <br> CBlock Cinder Block <br> None None <br> Stone Stone|
|**mas_vnr_area**|*float*|Masonry veneer area in square feet|square feet|
|**exter_qual**|*int*|Exterior material quality|Ex Excellent <br> Gd Good <br> TA Average/Typical <br> Fa Fair <br> Po Poor|
|**exter_cond**|*object*|Present condition of the material on the exterior|Ex Excellent <br> Gd Good <br> TA Average/Typical <br> Fa Fair <br> Po Poor|
|**foundation**|*int*|Type of foundation|BrkTil Brick & Tile <br> CBlock Cinder Block <br> PConc Poured Contrete <br> Slab Slab <br> Stone Stone <br> Wood Wood|
|**bsmt_qual**|*object*|Height of the basement|Ex Excellent (100+ inches) <br> Gd Good (90-99 inches) <br> TA Typical (80-89 inches) <br> Fa Fair (70-79 inches) <br> Po Poor (<70 inches) <br> NA No Basement|
|**bsmt_cond**|*object*|General condition of the basement|Ex Excellent <br> Gd Good <br> TA Typical - slight dampness allowed <br> Fa Fair - dampness or some cracking or settling <br> Po Poor - Severe cracking, settling, or wetness <br> NA No Basement|
|**bsmt_exposure**|*object*|Walkout or garden level basement walls|Gd Good Exposure <br> Av Average Exposure (split levels or foyers typically score average or above) <br> Mn Mimimum Exposure <br> No No Exposure <br> NA No Basement|
|**bsmtfin_type_1**|*object*|Quality of basement finished area|GLQ Good Living Quarters <br> ALQ Average Living Quarters <br> BLQ Below Average Living Quarters <br> Rec Average Rec Room <br> LwQ Low Quality <br> Unf Unfinshed <br> NA No Basement|
|**bsmtfin_sf_1**|*int*|Type 1 finished square feet|square feet|
|**bsmtfin_type_2**|*object*|Quality of second finished area (if present)|GLQ Good Living Quarters <br> ALQ Average Living Quarters <br> BLQ Below Average Living Quarters <br> Rec Average Rec Room <br> LwQ Low Quality <br> Unf Unfinshed <br> NA No Basement|
|**bsmtfin_sf_2**|*int*|Type 2 finished square feet|square feet|
|**bsmt_unf_sf**|*int*|Unfinished square feet of basement area|square feet|
|**total_bsmt_sf**|*int*|Total square feet of basement area|square feet|
|**heating**|*object*|Type of heating|Floor Floor Furnace <br> GasA Gas forced warm air furnace <br> GasW Gas hot water or steam heat <br> Grav Gravity furnace <br> OthW Hot water or steam heat other than gas <br> Wall Wall furnace|
|**heating_qc**|*object*|Heating quality and condition|Ex Excellent <br> Gd Good <br> TA Average/Typical <br> Fa Fair <br> Po Poor|
|**central_air**|*object*|Central air conditioning|N No <br> Y Yes|
|**electrical**|*object*|Electrical system|SBrkr Standard Circuit Breakers & Romex <br> FuseA Fuse Box over 60 AMP and all Romex wiring (Average) <br> FuseF 60 AMP Fuse Box and mostly Romex wiring (Fair) <br> FuseP 60 AMP Fuse Box and mostly knob & tube wiring (poor) <br> Mix Mixed|
|**1st_flr_sf**|*int*|First Floor square feet|square feet|
|**2nd_flr_sf**|*int*|Second Floor square feet|square feet|
|**low_qual_fin_sf**|*int*|Low quality finished square feet (all floors)|square feet|
|**gr_liv_area**|*int*|Above grade (ground) living area square feet|square feet|
|**bsmt_full_bath**|*float*|Basement full bathrooms|Number|
|**bsmt_half_bath**|*float*|Basement half bathrooms|Number|
|**full_bath**|*int*|Full bathrooms above grade|Integer|
|**galf_bath**|*int*|Half baths above grade|Integer|
|**bedroom_abvgr**|*int*|Number of bedrooms above basement level|Integer|
|**kitchen_abvgr**|*int*|Number of kitchens|Integer|
|**kitchen_qual**|*int*|Kitchen quality|Ex Excellent <br> Gd Good <br> TA Average/Typical <br> Fa Fair <br> Po Poor|
|**totrms_abvgrd**|*int*|Total rooms above grade (does not include bathrooms)|Integer|
|**functional**|*object*|Home functionality rating|Typ Typical Functionality <br> Min1 Minor Deductions 1 <br> Min2 Minor Deductions 2 Mod Moderate Deductions <br> Maj1 Major Deductions 1 <br> Maj2 Major Deductions 2 <br> Sev Severely Damaged <br> Sal Salvage only|
|**fireplaces**|*int*|umber of fireplaces|Integer|
|**fireplace_qu**|*object*|Fireplac quality|Ex Excellent - Exceptional Masonry Fireplace <br> Gd Good - Masonry Fireplace in main level <br> TA Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement <br> Fa Fair - Prefabricated Fireplace in basement <br> Po Poor - Ben Franklin Stove <br> NA No Fireplace|
|**garage_type**|*object*|Garage location|2Types More than one type of garage <br> Attchd Attached to home <br> Basment Basement Garage <br> BuiltIn Built-In (Garage part of house - typically has room above garage) <br> CarPort Car Port <br> Detchd Detached from home <br> NA No Garage|
|**garage_yr_blt**|*int*|Year garage was built|Year|
|**garage_finish**|*object*|Interior finish of the garage|Fin Finished <br> RFn Rough Finished <br> Unf Unfinished <br> NA No Garage|
|**garage_cars**|*float*|Size of garage in car capacity|Number|
|**garage_area**|*float*|Size of garage in square feet|square feet|
|**garage_qual**|*object*|Garage quality|Ex Excellent <br> Gd Good <br> TA Average/Typical <br> Fa Fair <br> NA No Garage|
|**garage_cond**|*object*|Garage condition|Ex Excellent <br> Gd Good <br> TA Average/Typical <br> Fa Fair <br> NA No Garage|
|**paved_drive**|*object*|Paved driveway|vY Paved <br> P Partial Pavement <br> N Dirt/Gravelals|
|**wood_deck_sf**|*int*|Wood deck area in square feet|square feet|
|**open_porch_sf**|*int*|Open porch area in square feet|square feet|
|**enclosed_porch**|*int*|Enclosed porch area in square feet|square feet|
|**3ssn_porch**|*int*|Three season porch area in square feet|square feet|
|**screen_porch**|*int*|Screen porch area in square feet|square feet|
|**pool_area**|*int*|Pool area in square feet||square feet|
|**pool_qc**|*object*|Pool quality|Ex Excellent <br> Gd Good <br> TA Average/Typical <br> Fa Fair <br> NA No Pool|
|**fence**|*object*|Fence quality|GdPrv Good Privacy <br> MnPrv Minimum Privacy <br> GdWo Good Wood <br> MnWw Minimum Wood/Wire <br> NA No Fence|
|**misc_feature**|*object*|Miscellaneous feature not covered in other categories|Elev Elevator <br> Gar2 2nd Garage (if not described in garage section) <br> Othr Other <br> Shed Shed (over 100 SF) <br> TenC Tennis Court <br> NA None|
|**misc_val**|*int*|$Value of miscellaneous feature|USD|
|**mo_sold**|*int*|Month Sold|Month|
|**yr_sold**|*int*|Sold|Year|
|**sale_type**|*object*|Type of sale|WD Warranty Deed - Conventional <br> CWD Warranty Deed - Cash <br> VWD Warranty Deed - VA Loan <br> New Home just constructed and sold <br> COD Court Officer Deed/Estate <br> Con Contract 15% Down payment regular terms <br> ConLw Contract Low Down payment and low interest <br> ConLI Contract Low Interest <br> ConLD Contract Low Down <br> Oth Other|
|**saleprice**|*int*|The property's sale price in dollars. |USD|


## Conclusions and Recommendations

After modeling, we have learnt that the top 5 features that have the strongest influence over sales price are:

1. Above Ground Living Area
2. Overall Quality 	
3. Basement Finish Surface Area
4. External Quality
5. Total Basement Surface Area

The two good neighbourhoods are: 

1. Northridge Heights
2. Stone Brook 

The 5 features that hurt the value of a home the most are: 

1. Masonry veneer type BrkFace
2. Ms_subclass 2 storey
2. Bedroom above ground
3. Ms subclass 1 storey
4. Neighborhoood CollgCr
5. Exterior 2nd Plywood


I don't think lasso model can generalize to other cities, as the main problem with lasso regression model itself is that when we have correlated variables, it retains only one variable and sets other correlated variables to zero. That will possibly lead to some loss of information resulting in lower accuracy in our model.

If we want to retain all of the features, we can consider to apply ridge regression, as it has similar score as lasso. However, ridge will shrink the coefficients and the model will remain complex and might lead to poor model performance. Hence, we can consider elastic net regression, which is basically a hybrid of ridge and lasso regression.

In this case, I chose lasso model coz it has the best score among the three regression models. If we want to develop a model that can be generalized to other cities, elastic net regression would be a better option as it will not omit some variables like lasso. 

