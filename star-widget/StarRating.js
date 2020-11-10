import React, { useState } from 'react';
import { FaStar } from 'react-icons/fa';

const StarRating = () => {
	const [rating, setRatingValue] = useState(null);
	const [hover, setHoverValue] = useState(null);

	return (
		<div>
			{[...Array(5)].map((star, i) => {

			const currValue = i + 1;
				return (
						<label>
					 		<input 
					 			type="radio" 
					 			name="rating"
					 			value={currValue}
					 			onClick={() => {
					 				setRatingValue(currValue);
					 			}}
					 		/>
						 	<FaStar
						 		className="ratingstar"
						 		size={100}
						 		color={currValue <= (rating || hover) ? "orange": "grey"}
					 			onMouseEnter={() => {
					 				setHoverValue(currValue);
					 			}}
					 			onMouseLeave={() => {
					 				setHoverValue(null)
					 			}}
					 		/>
					</label>
					);

				})

			}
		</div>
		);
};

export default StarRating;

