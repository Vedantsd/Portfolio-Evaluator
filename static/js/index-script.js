function searchStock(){
    let input=document.getElementById("search").value.toUpperCase();
    let items=document.getElementsByClassName("stock-item");
    for(let i=0;i<items.length;i++){
        let name=items[i].getElementsByClassName("stock-name")[0];
        if(name.innerHTML.toUpperCase().indexOf(input)>-1){
            items[i].style.display="";
        }
        else{
            items[i].style.display="none";
        }
    }
}
