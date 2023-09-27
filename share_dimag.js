/*
facebook: https://www.facebook.com/sharer.php?u=[post-url]
whp: https://api.whatsapp.com/send?text=[post-title] [post-url]
twitter: https://twitter.com/share?url=[post-url]&text=[post-title]&via=[via]&hashtags=[hashtags]

*/

const facebookBtn=document.querySelector(".facebook-btn");
const twitterBtn=document.querySelector(".twitter-btn");
const whatsappBtn=document.querySelector(".whatsapp-btn");

// console.log("Hello");
let postUrl=encodeURI(document.location.href);
let postTitle=encodeURI("Guys look for my spotify playlist similarity with...");

facebookBtn.setAttribute("href",`https://www.facebook.com/sharer.php?u=${postUrl}`);
whatsappBtn.setAttribute("href",`https://api.whatsapp.com/send?text=${postTitle} ${postUrl}`);
twitterBtn.setAttribute("href",`https://twitter.com/share?url=${postUrl}&text=${postTitle}`); 