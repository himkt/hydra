(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5701],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return p},kt:function(){return f}});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function o(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var l=a.createContext({}),c=function(e){var t=a.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):o(o({},t),e)),n},p=function(e){var t=c(e.components);return a.createElement(l.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},d=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,l=e.parentName,p=s(e,["components","mdxType","originalType","parentName"]),d=c(n),f=r,m=d["".concat(l,".").concat(f)]||d[f]||u[f]||i;return n?a.createElement(m,o(o({ref:t},p),{},{components:n})):a.createElement(m,o({ref:t},p))}));function f(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,o=new Array(i);o[0]=d;var s={};for(var l in t)hasOwnProperty.call(t,l)&&(s[l]=t[l]);s.originalType=e,s.mdxType="string"==typeof e?e:r,o[1]=s;for(var c=2;c<i;c++)o[c]=n[c];return a.createElement.apply(null,o)}return a.createElement.apply(null,n)}d.displayName="MDXCreateElement"},2279:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return l},metadata:function(){return c},toc:function(){return p},default:function(){return d}});var a=n(2122),r=n(9756),i=(n(7294),n(3905)),o=["components"],s={id:"overview",title:"Instantiating objects with Hydra",sidebar_label:"Overview"},l=void 0,c={unversionedId:"patterns/instantiate_objects/overview",id:"version-1.0/patterns/instantiate_objects/overview",isDocsHomePage:!1,title:"Instantiating objects with Hydra",description:"One of the best ways to drive different behavior in an application is to instantiate different implementations of an interface.",source:"@site/versioned_docs/version-1.0/patterns/instantiate_objects/overview.md",sourceDirName:"patterns/instantiate_objects",slug:"/patterns/instantiate_objects/overview",permalink:"/docs/1.0/patterns/instantiate_objects/overview",editUrl:"https://github.com/facebookresearch/hydra/edit/master/website/versioned_docs/version-1.0/patterns/instantiate_objects/overview.md",version:"1.0",lastUpdatedBy:"Jieru Hu",lastUpdatedAt:1629405728,formattedLastUpdatedAt:"8/19/2021",frontMatter:{id:"overview",title:"Instantiating objects with Hydra",sidebar_label:"Overview"},sidebar:"version-1.0/docs",previous:{title:"Config Store API",permalink:"/docs/1.0/tutorials/structured_config/config_store"},next:{title:"Config files example",permalink:"/docs/1.0/patterns/instantiate_objects/config_files"}},p=[],u={toc:p};function d(e){var t=e.components,n=(0,r.Z)(e,o);return(0,i.kt)("wrapper",(0,a.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("p",null,"One of the best ways to drive different behavior in an application is to instantiate different implementations of an interface.\nThe code using the instantiated object only knows the interface which remains constant, but the behavior\nis determined by the actual object instance."),(0,i.kt)("p",null,"Hydra provides ",(0,i.kt)("inlineCode",{parentName:"p"},"hydra.utils.call()")," (and its alias ",(0,i.kt)("inlineCode",{parentName:"p"},"hydra.utils.instantiate()"),") for instantiating objects and calling functions. Prefer ",(0,i.kt)("inlineCode",{parentName:"p"},"instantiate")," for creating objects and ",(0,i.kt)("inlineCode",{parentName:"p"},"call")," for invoking functions."),(0,i.kt)("p",null,"Call/instantiate supports:"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Class names : Call the ",(0,i.kt)("inlineCode",{parentName:"li"},"__init__")," method"),(0,i.kt)("li",{parentName:"ul"},"Callables like functions, static functions, class methods and objects")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},'def call(config: Any, *args: Any, **kwargs: Any) -> Any:\n    """\n    :param config: An object describing what to call and what params to use.\n                   Must have a _target_ field.\n    :param args: optional positional parameters pass-through\n    :param kwargs: optional named parameters pass-through\n    :return: the return value from the specified class or method\n    """\n    ...\n\n# Alias for call\ninstantiate = call\n')),(0,i.kt)("p",null,"The config passed to these functions must have a key called ",(0,i.kt)("inlineCode",{parentName:"p"},"_target_"),", with the value of a fully qualified class name, class method, static method or callable.",(0,i.kt)("br",{parentName:"p"}),"\n","Any additional parameters are passed as keyword arguments to the target."),(0,i.kt)("p",null,"For example, your application may have a User class that looks like this:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="user.py"',title:'"user.py"'},"class User:\n  name: str\n  age : int\n  \n  def __init__(self, name: str, age: int):\n    self.name = name\n    self.age = age\n")),(0,i.kt)("div",{className:"row"},(0,i.kt)("div",{className:"col col--6"},(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-yaml",metastring:'title="Config"',title:'"Config"'},"bond:\n  _target_: user.User\n  name: Bond\n  age: 7\n"))),(0,i.kt)("div",{className:"col col--6"},(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python",metastring:'title="Instantiation"',title:'"Instantiation"'},'user : User = instantiate(cfg.bond)\nassert isinstance(user, user.User)\nassert user.name == "Bond"\nassert user.age == 7\n')))),(0,i.kt)("p",null,"For convenience, instantiate/call returns ",(0,i.kt)("inlineCode",{parentName:"p"},"None")," when receiving ",(0,i.kt)("inlineCode",{parentName:"p"},"None")," as input."),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-python"},"assert instantiate(None) is None\n")))}d.isMDXComponent=!0}}]);